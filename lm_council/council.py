import json
import asyncio
import os
import random
import re

import aiohttp
# import instructor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import tqdm.asyncio
from aiolimiter import AsyncLimiter
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, HfFolder
from openai import AsyncOpenAI
from openai import AsyncOpenAI
from typing import List, Dict, Union, Optional

from lm_council.analysis.pairwise.affinity import get_affinity_df
from lm_council.analysis.pairwise.agreement import get_judge_agreement_map
from lm_council.analysis.pairwise.bradley_terry import bradley_terry_analysis
from lm_council.analysis.pairwise.explicit_win_rate import get_explicit_win_rates
from lm_council.analysis.pairwise.pairwise_utils import get_reference_llm
from lm_council.analysis.pairwise.separability import (
    analyze_rankings_separability_polarization,
)
from lm_council.analysis.rubric.affinity import get_affinity_matrices
from lm_council.analysis.rubric.agreement import get_judge_agreement
from lm_council.analysis.visualization import (
    plot_arena_hard_elo_stats,
    plot_direct_assessment_charts,
    plot_heatmap,
)
from lm_council.judging import PRESET_EVAL_CONFIGS
from lm_council.judging.config import EvaluationConfig
from lm_council.judging.prompt_builder import (
    LIKERT_PREBUILT_MAP,
    check_prompt_template_contains_all_placeholders,
)
from lm_council.structured_outputs import (
    PAIRWISE_COMPARISON_LABEL_MAP,
    create_dynamic_schema,
    get_pairwise_comparison_schema,
)


def process_pairwise_choice(raw_pairwise_choice: str) -> str:
    return raw_pairwise_choice.replace("[", "").replace("]", "")


class LanguageModelCouncil:

    def __init__(
        self,
        models: List[str],
        judge_models: Optional[List[str]] = None,
        eval_config: Optional[EvaluationConfig] = PRESET_EVAL_CONFIGS["default_rubric"],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        self.models = models
        self.eval_config = eval_config

        self.judge_models = judge_models
        # If no judge models are provided, use the same models for judging.
        if judge_models is None:
            # Use the same models for judging.
            self.judge_models = models

        # 1. Setup OpenRouter Client
        self.openrouter_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.openrouter_api_base = api_base or "https://openrouter.ai/api/v1"
        
        self.client_openrouter = AsyncOpenAI(
            base_url=self.openrouter_api_base,
            api_key=self.openrouter_api_key,
        )

        # 2. Setup Google Client
        self.google_api_key = google_api_key or os.getenv("GEMINI_API_KEY")
        self.client_google = None
        if self.google_api_key:
            self.client_google = AsyncOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=self.google_api_key,
            )

        # 3. Rate Limiter (OpenRouter specific for now)
        # We'll use this limiter primarily for OpenRouter. Google has its own limits but we'll share for simplicity or just limit OR.
        # For now, let's keep the existing logic for OpenRouter rate limits.
        max_calls = 100
        interval_seconds = 1

        if self.openrouter_api_key and "openrouter.ai" in self.openrouter_api_base:
            try:
                key_meta = requests.get(
                    "https://openrouter.ai/api/v1/auth/key",
                    headers={"Authorization": f"Bearer {self.openrouter_api_key}"},
                    timeout=10,
                ).json()["data"]["rate_limit"]

                max_calls = key_meta["requests"]
                if max_calls <= 0:
                     max_calls = 100
                
                interval_seconds = (
                    10
                    if key_meta["interval"].endswith("s")
                    else int(key_meta["interval"][:-1])
                )
            except Exception as e:
                print(f"Warning: Failed to fetch OpenRouter rate limits: {e}")

        self._limiter = AsyncLimiter(max_calls, interval_seconds)
        
        # Strict limiter for Google Free Tier (approx 10-15 RPM)
        # We'll set it to 1 request every 5 seconds to be safe.
        self._google_limiter = AsyncLimiter(1, 5)

    def _get_client_for_model(self, model: str) -> AsyncOpenAI:
        """Returns the appropriate client based on the model name."""
        if model.startswith("gemini-") and not model.startswith("google/"):
            if self.client_google:
                return self.client_google
            else:
                raise ValueError(f"Model {model} requires Google API Key (GEMINI_API_KEY) which is not set.")
        return self.client_openrouter

        # If we're doing pairwise_comparisons with fixed_reference_model(s), check that each of the
        # models in config.algorithm_config.reference_models is also included in our completion
        # requests.
        if (
            self.eval_config.type == "pairwise_comparison"
            and getattr(self.eval_config.config, "algorithm_type", None)
            == "fixed_reference_models"
        ):
            reference_models = set(
                self.eval_config.config.algorithm_config.reference_models
            )
            missing = list(reference_models - set(self.models))

            if missing:
                print(
                    f"The following reference models are specified in config.algorithm_config.reference_models but are not present in self.models: {missing}. Adding these model(s) to the council."
                )
                self.models.extend(missing)

        # List of all user prompts.
        self.user_prompts = []

        # List of all completions.
        self.completions = []

        # List of all judgments.
        self.judgments = []

    async def _call_with_retry(self, func, model: str):
        """Run API call with rate limiting and retries for 429 errors."""
        retries = 5
        delay = 5.0
        
        while True:
            try:
                if model.startswith("gemini-") or model.startswith("google/"):
                    async with self._google_limiter:
                        return await func()
                else:
                    async with self._limiter:
                        return await func()
            except Exception as e:
                error_str = str(e)
                if ("429" in error_str or "Resource exhausted" in error_str) and retries > 0:
                    # Try to parse retry delay from error message
                    # Example: "Please retry in 17.714540019s."
                    match = re.search(r"retry in (\d+\.?\d*)s", error_str)
                    if match:
                        wait_time = float(match.group(1)) + 1.0 # Add buffer
                        print(f"Rate limit hit for {model}. Waiting {wait_time:.2f}s as requested...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Rate limit hit for {model}. Retrying in {delay}s... (Error: {error_str[:100]}...)")
                        await asyncio.sleep(delay)
                        delay *= 2
                    
                    retries -= 1
                else:
                    raise e

    async def get_judge_rubric_structured_output(
        self,
        user_prompt: str,
        judge_prompt: str,
        judge_model: str,
        model_being_judged: str,
        schema_class: type,
    ) -> dict:
        """Get an async structured output task for the given prompt."""
        # Append JSON schema instructions to the prompt
        schema_json = schema_class.model_json_schema()
        judge_prompt += f"\n\nPlease respond with a valid JSON object matching this schema:\n{json.dumps(schema_json, indent=2)}"

        client = self._get_client_for_model(judge_model)
        completion = await self._call_with_retry(
            lambda: client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=self.eval_config.temperature,
                response_format={"type": "json_object"},
            ),
            model=judge_model
        )
        
        completion_text = completion.choices[0].message.content
        try:
            structured_output = schema_class.model_validate_json(completion_text)
        except Exception as e:
            print(f"Error parsing JSON from {judge_model}: {e}")
            print(f"Completion text: {completion_text}")
            # Fallback or re-raise? For now, re-raise to see what happens
            raise e

        return {
            "user_prompt": user_prompt,
            "judge_model": judge_model,
            "model_being_judged": model_being_judged,
            "structured_output": structured_output,
            "temperature": self.eval_config.temperature,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    async def get_text_completion(
        self,
        user_prompt: str,
        model: str,
        temperature: Optional[float] = None,
    ):
        """Get an async text completion task for the given prompt."""
        client = self._get_client_for_model(model)
        completion = await self._call_with_retry(
            lambda: client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            ),
            model=model
        )

        return {
            "user_prompt": user_prompt,
            "model": model,
            "completion_text": completion.choices[0].message.content,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    async def get_text_completions(
        self,
        user_prompt: str,
        temperature: Optional[float] = None,
    ):
        """Get an async text completion task for the given prompt."""
        return [
            self.get_text_completion(user_prompt, model, temperature)
            for model in self.models
        ]

    async def collect_completions(
        self, user_prompt: str, temperature: Optional[float] = None
    ) -> pd.DataFrame:
        tasks = await self.get_text_completions(
            user_prompt,
            temperature=temperature,
        )

        print(f"Generated {len(tasks)} completion tasks for user prompt: {user_prompt}")

        completions = []
        for future in tqdm.asyncio.tqdm.as_completed(tasks, total=len(tasks)):
            completions.append(await future)

        return pd.DataFrame(completions)

    async def get_judge_rubric_tasks(
        self,
        completions_df: pd.DataFrame,
        temperature: Optional[float] = None,
    ) -> List:
        judging_tasks = []
        for judge_model in self.judge_models:
            for _, row in completions_df.iterrows():
                model_being_judged = row["model"]

                # Skip if the judge model is the same as the model being judged.
                if (
                    self.eval_config.exclude_self_grading
                    and judge_model == model_being_judged
                ):
                    continue

                user_prompt = row["user_prompt"]

                criteria_verbalized = []
                for criteria in self.eval_config.config.rubric:
                    criteria_verbalized.append(f"{criteria.name}: {criteria.statement}")

                likert_scale_verbalized = LIKERT_PREBUILT_MAP[
                    self.eval_config.config.prebuilt_likert_scale
                ]

                # Get the judge prompt.
                judge_prompt = self.eval_config.config.prompt_template.format(
                    criteria_verbalized=criteria_verbalized,
                    likert_scale_verbalized=likert_scale_verbalized,
                    user_prompt=user_prompt,
                    response=row["completion_text"],
                )

                schema_class = create_dynamic_schema(self.eval_config)

                judging_tasks.append(
                    self.get_judge_rubric_structured_output(
                        user_prompt=user_prompt,
                        judge_prompt=judge_prompt,
                        judge_model=judge_model,
                        model_being_judged=model_being_judged,
                        schema_class=schema_class,
                    )
                )
        return judging_tasks

    async def get_judge_rubric_ratings(
        self,
        completions_df: pd.DataFrame,
        temperature: Optional[float] = None,
    ) -> pd.DataFrame:
        """Get an async structured output task for the given prompt."""
        judging_tasks = await self.get_judge_rubric_tasks(
            completions_df, temperature=temperature
        )

        print(f"Generated {len(judging_tasks)} judging tasks.")

        judgments = []
        for future in tqdm.asyncio.tqdm.as_completed(
            judging_tasks, total=len(judging_tasks)
        ):
            result = await future

            # Replace 'structured_output' with its attributes as columns
            structured_output = result.pop("structured_output")
            result.update(structured_output.model_dump())
            judgments.append(result)

        # Add an overall score column, which is the mean of all criteria scores.
        for judgment in judgments:
            criteria_scores = [
                judgment[f"{criteria.name}"]
                for criteria in self.eval_config.config.rubric
            ]
            judgment["Overall"] = sum(criteria_scores) / len(criteria_scores)
        return pd.DataFrame(judgments)

    async def get_judge_pairwise_structured_output(
        self,
        user_prompt: str,
        completions_map: Dict[str, str],
        llm1: str,
        llm2: str,
        judge_model: str,
        temperature: Optional[float] = None,
    ):
        prompt_template = self.eval_config.config.prompt_template
        prompt_template_fields = {
            "user_prompt": user_prompt,
            "response_1": completions_map[llm1],
            "response_2": completions_map[llm2],
            "pairwise_comparison_labels": PAIRWISE_COMPARISON_LABEL_MAP[
                self.eval_config.config.granularity
            ],
        }

        schema_class = get_pairwise_comparison_schema(
            self.eval_config.config.granularity,
            self.eval_config.cot_enabled,
        )

        check_prompt_template_contains_all_placeholders(
            prompt_template, prompt_template_fields
        )

        judge_prompt = prompt_template.format(**prompt_template_fields)

        # Append JSON schema instructions to the prompt
        schema_json = schema_class.model_json_schema()
        judge_prompt += f"\n\nPlease respond with a valid JSON object matching this schema:\n{json.dumps(schema_json, indent=2)}"

        client = self._get_client_for_model(judge_model)
        completion = await self._call_with_retry(
            lambda: client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            ),
            model=judge_model
        )
        
        completion_text = completion.choices[0].message.content
        try:
            structured_output = schema_class.model_validate_json(completion_text)
        except Exception as e:
            print(f"Error parsing JSON from {judge_model}: {e}")
            print(f"Completion text: {completion_text}")
            raise e

        return {
            "user_prompt": user_prompt,
            "judge_model": judge_model,
            "first_completion_by": llm1,
            "second_completion_by": llm2,
            "structured_output": structured_output,
            "temperature": temperature,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }

    async def get_judge_pairwise_rating_tasks_for_single_prompt(
        self,
        user_prompt: str,
        completions_df: pd.DataFrame,
        temperature: Optional[float] = None,
    ) -> List:
        completions_map = {
            row["model"]: row["completion_text"] for _, row in completions_df.iterrows()
        }

        pairwise_comparison_config = self.eval_config.config

        # Generate all pairs of completions.
        completion_pairs = [
            (llm1, llm2)
            for i, llm1 in enumerate(completions_map.keys())
            for llm2 in list(completions_map.keys())[i + 1 :]
        ]

        # Skip equal pairs.
        if pairwise_comparison_config.skip_equal_pairs:
            completion_pairs = [
                (llm1, llm2)
                for llm1, llm2 in completion_pairs
                if completions_map[llm1] != completions_map[llm2]
            ]

        # Filter down the pairs based on the pairwise_comparison_config.
        if pairwise_comparison_config.algorithm_type == "all_pairs":
            # No filtering needed.
            pass
        elif pairwise_comparison_config.algorithm_type == "random_pairs":
            # Generate a random sample of pairs of completions.
            completion_pairs = random.sample(
                completion_pairs,
                pairwise_comparison_config.n_random_pairs,
            )
        elif pairwise_comparison_config.algorithm_type == "fixed_reference_models":
            # Use llm1 as a fixed reference model.
            completion_pairs = [
                pair
                for pair in completion_pairs
                if pair[0]
                in pairwise_comparison_config.algorithm_config.reference_models
            ]

        # Apply positional flipping.
        # NOTE: This must happen after filtering down the pairs.
        if pairwise_comparison_config.position_flipping:
            completion_pairs = [
                (llm2, llm1) for llm1, llm2 in completion_pairs
            ] + completion_pairs

        # Apply reps.
        completion_pairs = [
            (llm1, llm2)
            for llm1, llm2 in completion_pairs
            for _ in range(self.eval_config.reps)
        ]

        # Fail if completion_pairs is empty.
        if len(completion_pairs) == 0:
            raise ValueError(
                "No pairs of completions to judge. Please check your pairwise_comparison_config. Perhaps reference models are misspelled?"
            )

        # Convert the completion_pairs into tasks.
        tasks = []
        for llm1, llm2 in completion_pairs:
            for judge_model in self.judge_models:
                if self.eval_config.exclude_self_grading and (
                    llm1 == judge_model or llm2 == judge_model
                ):
                    # Self-grading is disabled.
                    continue

                # Generate a judging task.
                tasks.append(
                    self.get_judge_pairwise_structured_output(
                        user_prompt=user_prompt,
                        completions_map=completions_map,
                        llm1=llm1,
                        llm2=llm2,
                        judge_model=judge_model,
                        temperature=temperature,
                    )
                )

        return tasks

    async def get_judge_pairwise_rating_tasks(
        self,
        completions_df: pd.DataFrame,
        temperature: Optional[float] = None,
    ) -> List:
        # Create a map of user_prompt -> model -> completion.
        user_prompts = completions_df["user_prompt"].unique()

        tasks = []
        for user_prompt in user_prompts:
            tasks.extend(
                await self.get_judge_pairwise_rating_tasks_for_single_prompt(
                    user_prompt=user_prompt,
                    completions_df=completions_df[
                        completions_df["user_prompt"] == user_prompt
                    ],
                    temperature=temperature,
                )
            )
        return tasks

    async def get_judge_pairwise_ratings(
        self,
        completions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        temperature = self.eval_config.temperature
        judging_tasks = await self.get_judge_pairwise_rating_tasks(
            completions_df,
        )

        print(f"Generated {len(judging_tasks)} judging tasks.")

        judgments = []
        for future in tqdm.asyncio.tqdm.as_completed(
            judging_tasks, total=len(judging_tasks)
        ):
            result = await future

            # Replace 'structured_output' with its attributes as columns
            structured_output = result.pop("structured_output")
            result.update(structured_output.model_dump())

            # Process the pairwise choice to remove brackets.
            result["pairwise_choice"] = process_pairwise_choice(
                result["pairwise_choice"]
            )
            judgments.append(result)

        return judgments

    async def judge(
        self,
        completions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.eval_config.type == "direct_assessment":
            judgments = await self.get_judge_rubric_ratings(completions_df)
        elif self.eval_config.type == "pairwise_comparison":
            judgments = await self.get_judge_pairwise_ratings(completions_df)
        else:
            raise ValueError(
                f"Invalid evaluation config type: {self.eval_config.type}. Must be one of: direct_assessment, pairwise_comparison."
            )
        return pd.DataFrame(judgments)

    def get_judging_df(self) -> pd.DataFrame:
        """Returns the judgments made by the council."""
        return pd.DataFrame(self.judgments)

    def get_completions_df(self) -> pd.DataFrame:
        """Returns the completions made by the council."""
        return pd.DataFrame(self.completions)

    async def execute(self, prompts: Union[str, List[str]]):
        # Normalize to list[str]
        if isinstance(prompts, str):
            prompts = [prompts]

        # ─── 1️⃣  completions phase ─────────
        completion_tasks = [
            task
            for p in prompts
            for task in await self.get_text_completions(p, self.eval_config.temperature)
        ]

        print(f"Generated {len(completion_tasks)} completion tasks.")

        completions_raw = [
            await fut
            for fut in tqdm.asyncio.tqdm.as_completed(
                completion_tasks, total=len(completion_tasks)
            )
        ]
        completions_df = pd.DataFrame(completions_raw)
        self.completions.extend(completions_raw)
        self.user_prompts.extend(prompts)

        # ─── 2️⃣  judging phase ─────────
        judging_df = await self.judge(
            completions_df=completions_df,
        )
        self.judgments.extend(judging_df.to_dict("records"))
        return completions_df, judging_df

    def leaderboard(self, outfile: Optional[str] = None) -> pd.DataFrame:
        if self.eval_config.type == "pairwise_comparison":
            judging_df = self.get_judging_df()
            reference_llm = get_reference_llm(
                judging_df,
                self.eval_config,
            )

            rankings_results = analyze_rankings_separability_polarization(
                judging_df,
                reference_llm_respondent=reference_llm,
                bootstrap_rounds=10,
                include_individual_judges=True,
                include_council_majority=True,
                include_council_mean_pooling=True,
                include_council_no_aggregation=True,
                example_id_column="user_prompt",
            )

            if outfile:
                show = False
            else:
                show = True
            plot_arena_hard_elo_stats(
                rankings_results["council/no-aggregation"]["elo_scores"],
                "",
                outfile=outfile,
                show=show,
            )
            return rankings_results
        elif self.eval_config.type == "direct_assessment":
            return plot_direct_assessment_charts(
                self.get_judging_df(), self.eval_config, outfile=outfile
            )
        else:
            raise ValueError(
                f"Unimplemented leaderboard for evaluation config type: {self.eval_config.type}. Must be one of: direct_assessment, pairwise_comparison."
            )

    def win_rate_heatmap(self) -> pd.DataFrame:
        if self.eval_config.type != "pairwise_comparison":
            raise ValueError(
                "Win rate heatmap can only be generated for pairwise comparison evaluations."
            )
        return bradley_terry_analysis(self.get_judging_df())

    def explicit_win_rate_heatmap(self):
        if self.eval_config.type != "pairwise_comparison":
            raise ValueError(
                "Explicit win rate heatmap can only be generated for pairwise comparison evaluations."
            )

        judging_df = self.get_judging_df()
        win_rate_map = get_explicit_win_rates(judging_df)
        num_models = len(self.models)
        figsize = (max(5, num_models), max(4, num_models))

        plot_heatmap(
            win_rate_map,
            ylabel="Respondent",
            xlabel="vs. Respondent",
            vmin=0,
            vmax=1,
            cmap="coolwarm",
            outfile=None,
            figsize=figsize,
            font_size=8,
        )

        return win_rate_map

    def judge_agreement(self, show_plots=True):
        if self.eval_config.type == "direct_assessment":
            judging_df = self.get_judging_df()

            agreement_matrices, mean_agreement_df = get_judge_agreement(
                judging_df, self.eval_config
            )

            if show_plots:
                # Plot agreement matrices for each criterion
                for crit_col, matrix in agreement_matrices.items():
                    plot_heatmap(
                        matrix,
                        ylabel="Judge Model",
                        xlabel="Judge Model",
                        title=f"Judge Agreement Matrix: {crit_col}",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm_r",
                        outfile=None,
                        figsize=(8, 6),
                        font_size=8,
                    )

            return agreement_matrices, mean_agreement_df

        elif self.eval_config.type == "pairwise_comparison":
            judge_agreement_map, mean_agreement_df = get_judge_agreement_map(
                self.get_judging_df(), example_id_column="user_prompt"
            )
            if show_plots:
                # Plot the judge agreement map as a heatmap
                for judge_model, agreement_matrix in judge_agreement_map.items():
                    plot_heatmap(
                        agreement_matrix,
                        ylabel="Judge Model",
                        xlabel="Judge Model",
                        title=f"Pairwise Judge Agreement: {judge_model}",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(8, 6),
                        font_size=8,
                    )
            return judge_agreement_map, mean_agreement_df

    def affinity(self, show_plots=True):
        if self.eval_config.type == "direct_assessment":
            affinity_matrices = get_affinity_matrices(
                self.get_judging_df(), self.eval_config
            )
            judge_models = self.judge_models
            models_being_judged = self.models

            if show_plots:
                for crit, matrix in affinity_matrices.items():
                    # Plot heatmap
                    plot_heatmap(
                        matrix,
                        ylabel="Model Being Judged",
                        xlabel="Judge Model",
                        title=f"Affinity Heatmap: {crit}",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(
                            max(8, len(judge_models) + 2),
                            max(6, len(models_being_judged)),
                        ),
                        font_size=8,
                    )

            return affinity_matrices
        elif self.eval_config.type == "pairwise_comparison":
            judging_df = self.get_judging_df()
            reference_llm = get_reference_llm(
                judging_df,
                self.eval_config,
            )
            affinity_results = get_affinity_df(
                judging_df,
                reference_llm_respondent=reference_llm,
                example_id_column="user_prompt",
            )
            if show_plots:
                if "judge_preferences" in affinity_results:
                    plot_heatmap(
                        affinity_results["judge_preferences"],
                        ylabel="Judge Model",
                        xlabel="Model Being Judged",
                        title="Judge Preferences",
                        vmin=0,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(
                            max(8, len(self.judge_models) + 2),
                            max(6, len(self.models)),
                        ),
                        font_size=8,
                    )
                if "judge_preferences_council_normalized" in affinity_results:
                    plot_heatmap(
                        affinity_results["judge_preferences_council_normalized"],
                        ylabel="Judge Model",
                        xlabel="Model Being Judged",
                        title="Judge Preferences (Council Normalized)",
                        vmin=-1,
                        vmax=1,
                        cmap="coolwarm",
                        outfile=None,
                        figsize=(
                            max(8, len(self.judge_models) + 2),
                            max(6, len(self.models)),
                        ),
                        font_size=8,
                    )
            return affinity_results

    def generate_hf_readme(self) -> str:
        """
        Generate a markdown string describing the dataset for Hugging Face Hub.
        """

        def make_serializable(obj):
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            elif hasattr(obj, "__dict__"):
                return {k: make_serializable(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            else:
                return obj

        eval_config_str = json.dumps(
            make_serializable(self.eval_config),
            indent=2,
        )
        num_prompts = len(set(self.user_prompts))
        models_judged = "\n- " + "\n- ".join(self.models)
        judge_models = (
            "\n- " + "\n- ".join(self.judge_models)
            if hasattr(self, "judge_models") and self.judge_models
            else models_judged
        )

        readme = f"""
## Leaderboard

![Leaderboard](leaderboard.png)

## Dataset Overview

**Number of unique prompts:** {num_prompts}

**Models evaluated:** {models_judged}

**Judge models:** {judge_models}

**Provider:** [OpenRouter](https://openrouter.ai)

## Evaluation Configuration
```json
{eval_config_str}
```

## About

This dataset was generated using the [LLM Council](https://github.com/llm-council/lm-council), a framework for evaluating language models by having them judge each other democratically.
"""

        return readme

    def upload_to_hf(self, repo_id: str):
        """Upload completions, judgments, leaderboard figure, and README to Hugging Face Hub as a dataset."""

        # Prepare datasets
        completions_ds = Dataset.from_pandas(
            pd.DataFrame(self.completions), preserve_index=False
        )
        judgments_ds = Dataset.from_pandas(
            pd.DataFrame(self.judgments), preserve_index=False
        )

        # Generate and save leaderboard figure
        leaderboard_df = self.leaderboard("leaderboard.png")

        leaderboard_ds = Dataset.from_pandas(
            leaderboard_df,
            preserve_index=False,
        )

        # Push each dataset to Hugging Face Hub with a config_name and split
        completions_ds.push_to_hub(repo_id, config_name="completions")
        judgments_ds.push_to_hub(repo_id, config_name="judgments")
        leaderboard_ds.push_to_hub(repo_id, config_name="leaderboard")

        # Upload leaderboard.png to the repo
        api = HfApi()
        api.upload_file(
            path_or_fileobj="leaderboard.png",
            path_in_repo="leaderboard.png",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add leaderboard figure",
        )
        os.remove("leaderboard.png")

        # Push README to the Hugging Face Hub
        readme_str = self.generate_hf_readme()
        # Get the current README if it exists
        try:
            old_readme = api.hf_hub_download(repo_id, "README.md", repo_type="dataset")
            with open(old_readme, "r", encoding="utf-8") as f:
                current_readme = f.read()
        except Exception:
            current_readme = ""

        # Append the new readme content
        combined_readme = current_readme + "\n\n" + readme_str

        api.upload_file(
            path_or_fileobj=combined_readme.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Append to README.md",
        )

    def save(self, outdir):
        # Save all artifacts to a directory.
        """
        outdir/
            models.json
            completions.jsonl
            judge_ratings.jsonl
            user_prompts.jsonl
            eval_config.json
        """
        os.makedirs(outdir, exist_ok=True)

        with open(os.path.join(outdir, "models.json"), "w") as f:
            json.dump(self.models, f)
        with open(os.path.join(outdir, "user_prompts.json"), "w") as f:
            json.dump(self.user_prompts, f)

        pd.DataFrame(self.completions).to_json(
            os.path.join(outdir, "completions.jsonl"), orient="records", lines=True
        )
        pd.DataFrame(self.judgments).to_json(
            os.path.join(outdir, "judge_ratings.jsonl"), orient="records", lines=True
        )
        self.eval_config.save_config(os.path.join(outdir, "eval_config.json"))

    @staticmethod
    def load(indir: str) -> "LanguageModelCouncil":
        """
        Load a LanguageModelCouncil instance from a directory.

        indir/
            models.json
            completions.jsonl
            judge_ratings.jsonl
            user_prompts.jsonl
            eval_config.json
        """
        # If indir is not an absolute path, make it relative to the current working directory
        if not os.path.isabs(indir):
            indir = os.path.abspath(os.path.join(os.getcwd(), indir))

        # Load LLMs
        with open(os.path.join(indir, "models.json"), "r") as f:
            models = json.load(f)

        # Load prompts
        with open(os.path.join(indir, "user_prompts.json"), "r") as f:
            user_prompts = json.load(f)

        # Load completions
        completions = pd.read_json(
            os.path.join(indir, "completions.jsonl"), orient="records", lines=True
        ).to_dict(orient="records")

        # Load judgments
        judgments = pd.read_json(
            os.path.join(indir, "judge_ratings.jsonl"), orient="records", lines=True
        ).to_dict(orient="records")

        # Load evaluation config
        eval_config = EvaluationConfig.load_config(
            os.path.join(indir, "eval_config.json")
        )

        # Create the instance
        council = LanguageModelCouncil(models=models, eval_config=eval_config)
        council.user_prompts = user_prompts
        council.completions = completions
        council.judgments = judgments

        return council

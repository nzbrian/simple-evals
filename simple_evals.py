import argparse
import json
import os
import tempfile
from datetime import datetime

import pandas as pd

from . import common
from .browsecomp_eval import BrowseCompEval
from .drop_eval import DropEval
from .gpqa_eval import GPQAEval
from .healthbench_eval import HealthBenchEval
from .healthbench_meta_eval import HealthBenchMetaEval
from .humaneval_eval import HumanEval
from .math_eval import MathEval
from .mgsm_eval import MGSMEval
from .mmlu_eval import MMLUEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.claude_sampler import CLAUDE_SYSTEM_MESSAGE_LMSYS, ClaudeCompletionSampler
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler
from .simpleqa_eval import SimpleQAEval


def main():
    parser = argparse.ArgumentParser(
        description="Run sampling and evaluations using different samplers and evaluations."
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List available models"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Select a model by name. Also accepts a comma-separated list of models.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        help="Select an eval by name. Also accepts a comma-separated list of evals.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats to run. Only supported for certain evals.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=120,
        help="Number of threads to run. Only supported for HealthBench and HealthBenchMeta.",
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=(
            "Base URL of an OpenAI-compatible API endpoint, e.g. "
            "http://localhost:8000/v1 for a local vLLM server. Defaults to the "
            "OPENAI_BASE_URL environment variable, then to the official OpenAI API. "
            "When set, the model under test and all grading models use this endpoint."
        ),
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help=(
            "API key for the endpoint. Defaults to the OPENAI_API_KEY environment "
            "variable, or 'dummy' (for local servers that do not require authentication)."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Model ID to send to the API (e.g. 'meta-llama/Llama-3.1-8B-Instruct'). "
            "When set, --model names the label used in result files and any value is "
            "accepted, so an arbitrary model served at --base-url can be tested. "
            "Without --model, the model ID is used as the label."
        ),
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default=None,
        help=(
            "Model ID used for LLM grading (SimpleQA, BrowseComp, HealthBench) and "
            "math equality checking. Defaults to the model under test when a custom "
            "--base-url is in use, otherwise gpt-4.1-2025-04-14."
        ),
    )
    parser.add_argument(
        "--no-system-message",
        action="store_true",
        help=(
            "Do not prepend the default OpenAI system message to the model under "
            "test (useful for local models that ignore or reject it)."
        ),
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Disable thinking/reasoning for the model under test by sending "
            "chat_template_kwargs={enable_thinking: false}. Needed for reasoning "
            "models (Qwen3, DeepSeek-R1, ...) served by llama.cpp/vLLM, which "
            "otherwise put their output in reasoning_content and leave content empty."
        ),
    )
    parser.add_argument(
        "--healthbench-input-path",
        type=str,
        default=None,
        help=(
            "Run the main HealthBench eval from a blobfile-readable JSONL path in "
            "HealthBench format. This only applies to --eval=healthbench."
        ),
    )
    parser.add_argument(
        "--healthbench-professional-mode",
        action="store_true",
        help=(
            "Require the HealthBench Professional option bundle: main HealthBench "
            "custom input path, GPT-5.4 low grader, and length adjustment."
        ),
    )
    parser.add_argument(
        "--healthbench-use-gpt-5-4-low-grader",
        action="store_true",
        help="Use GPT-5.4 with low reasoning effort as the HealthBench rubric grader.",
    )
    parser.add_argument(
        "--healthbench-length-adjustment-center",
        type=float,
        default=None,
        help=(
            "Center character count for HealthBench length adjustment. Must be set "
            "together with --healthbench-length-adjustment-penalty-per-500-chars."
        ),
    )
    parser.add_argument(
        "--healthbench-length-adjustment-penalty-per-500-chars",
        type=float,
        default=None,
        help=(
            "Numerical score penalty per 500 response characters for HealthBench "
            "length adjustment. Must be set together with "
            "--healthbench-length-adjustment-center."
        ),
    )

    args = parser.parse_args()
    if (
        args.healthbench_input_path is not None
        and args.eval is not None
        and "healthbench" not in args.eval.split(",")
    ):
        parser.error(
            "--healthbench-input-path only applies when --eval includes healthbench"
        )
    if (
        args.healthbench_length_adjustment_center is None
    ) != (args.healthbench_length_adjustment_penalty_per_500_chars is None):
        parser.error(
            "--healthbench-length-adjustment-center and "
            "--healthbench-length-adjustment-penalty-per-500-chars must be set together"
        )
    if args.healthbench_professional_mode:
        evals_requested = set(args.eval.split(",")) if args.eval is not None else set()
        if evals_requested != {"healthbench"}:
            parser.error(
                "--healthbench-professional-mode requires --eval=healthbench"
            )
        if args.healthbench_input_path is None:
            parser.error(
                "--healthbench-professional-mode requires --healthbench-input-path"
            )
        if not args.healthbench_use_gpt_5_4_low_grader:
            parser.error(
                "--healthbench-professional-mode requires "
                "--healthbench-use-gpt-5-4-low-grader"
            )
        if (
            args.healthbench_length_adjustment_center is None
            or args.healthbench_length_adjustment_penalty_per_500_chars is None
        ):
            parser.error(
                "--healthbench-professional-mode requires both HealthBench length "
                "adjustment flags"
            )

    # Endpoint configuration: any OpenAI-compatible API can be tested by
    # passing --base-url / --api-key (or setting OPENAI_BASE_URL /
    # OPENAI_API_KEY). The resolved values are threaded through every sampler
    # so a single endpoint retargets the model under test AND the grading
    # models, enabling fully self-contained runs against one server.
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
    if base_url:
        print(f"Using OpenAI-compatible endpoint: {base_url}")

    def _chat(
        model: str,
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ) -> ChatCompletionSampler:
        return ChatCompletionSampler(
            model=model,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            api_key=api_key,
        )

    def _o(model: str, reasoning_effort: str | None = None) -> OChatCompletionSampler:
        return OChatCompletionSampler(
            model=model,
            reasoning_effort=reasoning_effort,
            base_url=base_url,
            api_key=api_key,
        )

    def _resp(model: str, **kwargs) -> ResponsesSampler:
        return ResponsesSampler(
            model=model, base_url=base_url, api_key=api_key, **kwargs
        )

    models = {
        # Reasoning Models
        "o3": _resp(
            model="o3-2025-04-16",
            reasoning_model=True,
        ),
        "o3-temp-1": _resp(
            model="o3-2025-04-16",
            reasoning_model=True,
            temperature=1.0,
        ),
        "o3_high": _resp(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o3_low": _resp(
            model="o3-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        # Default == Medium
        "o4-mini": _resp(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
        ),
        "o4-mini_high": _resp(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="high",
        ),
        "o4-mini_low": _resp(
            model="o4-mini-2025-04-16",
            reasoning_model=True,
            reasoning_effort="low",
        ),
        "o1-pro": _resp(
            model="o1-pro",
            reasoning_model=True,
        ),
        "o1": _o(
            model="o1",
        ),
        "o1_high": _o(
            model="o1",
            reasoning_effort="high",
        ),
        "o1_low": _o(
            model="o1",
            reasoning_effort="low",
        ),
        "o1-preview": _o(
            model="o1-preview",
        ),
        "o1-mini": _o(
            model="o1-mini",
        ),
        # Default == Medium
        "o3-mini": _o(
            model="o3-mini",
        ),
        "o3-mini_high": _o(
            model="o3-mini",
            reasoning_effort="high",
        ),
        "o3-mini_low": _o(
            model="o3-mini",
            reasoning_effort="low",
        ),
        # GPT-4.1 models
        "gpt-4.1": _chat(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-temp-1": _chat(
            model="gpt-4.1-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4.1-mini": _chat(
            model="gpt-4.1-mini-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4.1-nano": _chat(
            model="gpt-4.1-nano-2025-04-14",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4o models
        "gpt-4o": _chat(
            model="gpt-4o",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-11-20": _chat(
            model="gpt-4o-2024-11-20",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06": _chat(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-2024-08-06-temp-1": _chat(
            model="gpt-4o-2024-08-06",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
            temperature=1.0,
        ),
        "gpt-4o-2024-05-13": _chat(
            model="gpt-4o-2024-05-13",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        "gpt-4o-mini": _chat(
            model="gpt-4o-mini-2024-07-18",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4.5 model
        "gpt-4.5-preview": _chat(
            model="gpt-4.5-preview-2025-02-27",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            max_tokens=2048,
        ),
        # GPT-4-turbo model
        "gpt-4-turbo-2024-04-09": _chat(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-4 model
        "gpt-4-0613": _chat(
            model="gpt-4-0613",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        # GPT-3.5 Turbo model
        "gpt-3.5-turbo-0125": _chat(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
        ),
        "gpt-3.5-turbo-0125-temp-1": _chat(
            model="gpt-3.5-turbo-0125",
            system_message=OPENAI_SYSTEM_MESSAGE_API,
            temperature=1.0,
        ),
        # Chatgpt models:
        "chatgpt-4o-latest": _chat(
            model="chatgpt-4o-latest",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
            max_tokens=2048,
        ),
        "gpt-4-turbo-2024-04-09_chatgpt": _chat(
            model="gpt-4-turbo-2024-04-09",
            system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
        ),
        # Claude models:
        "claude-3-opus-20240229_empty": ClaudeCompletionSampler(
            model="claude-3-opus-20240229",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-7-sonnet-20250219": ClaudeCompletionSampler(
            model="claude-3-7-sonnet-20250219",
            system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
        ),
        "claude-3-haiku-20240307": ClaudeCompletionSampler(
            model="claude-3-haiku-20240307",
        ),
    }

    if args.list_models:
        print("Available models:")
        for model_name in models:
            print(f" - {model_name}")
        return

    if args.model or args.model_name:
        if args.model_name:
            # Custom model: test an arbitrary model ID served at the endpoint.
            if args.model:
                models_chosen = args.model.split(",")
                if len(models_chosen) > 1:
                    parser.error(
                        "--model-name can only be combined with a single --model label"
                    )
                label = models_chosen[0]
            else:
                label = args.model_name
            system_message = (
                None if args.no_system_message else OPENAI_SYSTEM_MESSAGE_API
            )
            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if args.no_thinking
                else None
            )
            models = {
                label: ChatCompletionSampler(
                    model=args.model_name,
                    system_message=system_message,
                    base_url=base_url,
                    api_key=api_key,
                    extra_body=extra_body,
                )
            }
        else:
            models_chosen = args.model.split(",")
            for model_name in models_chosen:
                if model_name not in models:
                    print(
                        f"Error: Model '{model_name}' not found. "
                        "To test an arbitrary model ID served at an OpenAI-compatible "
                        "endpoint, use --model-name (and --base-url)."
                    )
                    return
            models = {model_name: models[model_name] for model_name in models_chosen}

    print(f"Running with args {args}")

    # Grading models also use the selected endpoint so a run can be fully
    # self-contained against a single OpenAI-compatible API.
    if args.grader_model:
        grader_model_id = args.grader_model
    elif base_url:
        if len(models) != 1:
            parser.error(
                "With a custom --base-url and multiple models, --grader-model is required"
            )
        grader_model_id = next(iter(models.values())).model
    else:
        grader_model_id = "gpt-4.1-2025-04-14"
    grading_sampler = ChatCompletionSampler(
        model=grader_model_id,
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url=base_url,
        api_key=api_key,
    )
    healthbench_grading_sampler = (
        ResponsesSampler(
            model="gpt-5.4-2026-03-05",
            reasoning_model=True,
            reasoning_effort="low",
            base_url=base_url,
            api_key=api_key,
        )
        if args.healthbench_use_gpt_5_4_low_grader
        else grading_sampler
    )
    equality_model_id = (
        args.grader_model
        if args.grader_model
        else (grader_model_id if base_url else "gpt-4-turbo-preview")
    )
    equality_checker = ChatCompletionSampler(
        model=equality_model_id,
        base_url=base_url,
        api_key=api_key,
    )
    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "mmlu":
                return MMLUEval(num_examples=1 if debug_mode else num_examples)
            case "math":
                return MathEval(
                    equality_checker=equality_checker,
                    num_examples=num_examples,
                    n_repeats=1 if debug_mode else args.n_repeats or 10,
                )
            case "gpqa":
                return GPQAEval(
                    n_repeats=1 if debug_mode else args.n_repeats or 10,
                    num_examples=num_examples,
                )
            case "mgsm":
                return MGSMEval(
                    num_examples_per_lang=10 if debug_mode else num_examples or 250
                )
            case "drop":
                return DropEval(
                    num_examples=10 if debug_mode else num_examples,
                    train_samples_per_prompt=3,
                )
            case "humaneval":
                return HumanEval(num_examples=10 if debug_mode else num_examples or 250)
            case "simpleqa":
                return SimpleQAEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "browsecomp":
                return BrowseCompEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                )
            case "healthbench":
                return HealthBenchEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                    input_path=args.healthbench_input_path,
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=(
                        args.healthbench_length_adjustment_penalty_per_500_chars
                    ),
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=(
                        args.healthbench_length_adjustment_penalty_per_500_chars
                    ),
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=healthbench_grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                    length_adjustment_center=args.healthbench_length_adjustment_center,
                    length_adjustment_penalty_per_500_chars=(
                        args.healthbench_length_adjustment_penalty_per_500_chars
                    ),
                )
            case "healthbench_meta":
                return HealthBenchMetaEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=args.n_repeats or 1,
                    n_threads=args.n_threads or 1,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    if args.eval:
        evals_list = args.eval.split(",")
        evals = {}
        for eval_name in evals_list:
            try:
                evals[eval_name] = get_evals(eval_name, args.debug)
            except Exception:
                print(f"Error: eval '{eval_name}' not found.")
                return
    else:
        evals = {
            eval_name: get_evals(eval_name, args.debug)
            for eval_name in [
                "mmlu",
                "math",
                "gpqa",
                "mgsm",
                "drop",
                "humaneval",
                "simpleqa",
                "browsecomp",
                "healthbench",
                "healthbench_hard",
                "healthbench_consensus",
                "healthbench_meta",
            ]
        }

    print(evals)
    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {list(evals.keys())}")
    print(f"Running evals for the following models: {list(models.keys())}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"{tempfile.gettempdir()}/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"{tempfile.gettempdir()}/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = f"{tempfile.gettempdir()}/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            with open(result_filename, "r+") as rf:
                result = json.load(rf)
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["model_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()

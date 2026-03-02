"""
Pairwise Summarization Evaluation
Candidate   : OpenAI (GPT-5.2)
Baseline    : Google Gemini via Vertex AI (gemini-3.1-pro-preview)
Judge       : Vertex AI Judge Model (gemini-2.5-pro)
Gemini Auth : Google Application Default
OpenAI Auth : API Key (env var)

Available Vertex AI Pairwise Metrics:
  - pairwise_summarization_quality      : Compares holistic quality, accuracy, and completeness of two summaries.
  - pairwise_question_answering_quality : Evaluates which response provides a better, more accurate answer to a question.
  - pairwise_helpfulness                : Evaluates which response is more directly helpful and addresses the user's goal.
  - pairwise_groundedness               : Assesses which response is more faithful to the source text (less hallucination).
  - pairwise_instruction_following      : Compares how accurately the models followed specific constraints/instructions.
  - pairwise_verbosity                  : Evaluates which response is appropriately concise without unnecessary padding.
  - pairwise_safety                     : Compares which response better avoids harmful, toxic, or unsafe content.

Dataset:
  10x_evaluation_dataset_001.csv

Prerequisites:
  pip install "google-cloud-aiplatform[evaluation]>=1.114.0" openai pandas google-genai

Authentication:
  gcloud auth application-default login
  gcloud config set project YOUR_PROJECT_ID

Required environment variable:
  OPENAI_API_KEY   — your OpenAI API key
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import datetime
import pandas as pd
import vertexai

from openai import OpenAI
from google import genai
from google.genai import types
from google.cloud import aiplatform

# Because AutoraterConfig is a preview feature, all associated evaluation 
# classes must be imported from the preview module to ensure compatibility.
from vertexai.preview.evaluation import (
    EvalTask,
    PairwiseMetric,
    MetricPromptTemplateExamples,
    AutoraterConfig,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Configuration
# ─────────────────────────────────────────────────────────────────────────────

# Google Cloud settings
PROJECT_ID          = os.environ.get("GCP_PROJECT_ID")
EVALUATION_LOCATION = "us-central1"
MODEL_LOCATION      = "global" # gemini 3.0/3.1. in preview, must be global
EXPERIMENT          = "summarization-pairwise-eval"   # lowercase/hyphens only

# Model identifiers
OPENAI_MODEL    = "gpt-5.2"
GEMINI_MODEL_ID = "gemini-3.1-pro-preview"

# Dataset path — CSV sits alongside this script
DATASET_PATH = "10x_evaluation_dataset_001.csv"
OUTPUT_DIR   = "model_output" # where output csv will be wrote to

# -----------------------------------------------------------------------------
# Metric Configuration (Enable/Disable metrics to evaluate)
# -----------------------------------------------------------------------------
EVALUATION_METRICS = {
    "pairwise_summarization_quality": True,
    "pairwise_verbosity": True,
    "pairwise_groundedness": True,
    "pairwise_question_answering_quality": False,
    "pairwise_helpfulness": False,
    "pairwise_instruction_following": False,
    "pairwise_safety": False,
}

# -----------------------------------------------------------------------------
# LLM Parameters Configuration (Candidate & Baseline)
# -----------------------------------------------------------------------------
# OpenAI Generation Parameters
OPENAI_PARAMS = {
    "temperature": 1.0,
    "max_completion_tokens": 10000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Gemini Generation Parameters
GEMINI_PARAMS = {
    "temperature": 1.0,
    "max_output_tokens": 10000,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Auto-configure thinking parameters based on the Gemini model version
# Gemini thinking model changed between 2.5 --> 3+
if "gemini-3" in GEMINI_MODEL_ID:
    # Gemini 3+ models support "Thinking level" (minimal, low, medium, high)
    GEMINI_PARAMS["thinking_level"] = "high"
elif "gemini-2.5" in GEMINI_MODEL_ID:
    # Gemini 2.5 models support "Thinking mode" boolean (enables/disables budget)
    GEMINI_PARAMS["thinking_mode"] = True

# -----------------------------------------------------------------------------
# Judge Model Configuration (Vertex AI AutoRater)
# -----------------------------------------------------------------------------
JUDGE_PARAMS = {
    "autorater_model": f"projects/{PROJECT_ID}/locations/{EVALUATION_LOCATION}/publishers/google/models/gemini-2.5-pro",  
    "sampling_count": 3,        # Multi-sampling (votes) to eliminate judge randomness (1 to 32)
    "flip_enabled": True,       # Prevent positional bias by flipping Model A and B
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Client Initialisation
# ─────────────────────────────────────────────────────────────────────────────

# 1. Initialize Vertex AI explicitly for the evaluation & experiments endpoint
vertexai.init(project=PROJECT_ID, location=EVALUATION_LOCATION)

# 2. Initialize OpenAI for the candidate model
openai_client = OpenAI()

# 3. Initialize the new Google GenAI SDK for the baseline model endpoint
gemini_client = genai.Client(
    vertexai=True, 
    project=PROJECT_ID, 
    location=MODEL_LOCATION
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Load Evaluation Dataset
# ─────────────────────────────────────────────────────────────────────────────
eval_dataset = pd.read_csv(DATASET_PATH)

print(f"✓ Dataset loaded: {len(eval_dataset)} rows from '{DATASET_PATH}'")
print(f"  Columns: {list(eval_dataset.columns)}\n")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Model Callables
# ─────────────────────────────────────────────────────────────────────────────

def gpt52_model(prompt: str) -> str:
    """Call GPT-5.2 via OpenAI API using configured parameters."""
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=OPENAI_PARAMS["temperature"],
        max_completion_tokens=OPENAI_PARAMS["max_completion_tokens"],
        top_p=OPENAI_PARAMS["top_p"],
        frequency_penalty=OPENAI_PARAMS["frequency_penalty"],
        presence_penalty=OPENAI_PARAMS["presence_penalty"],
    )
    return response.choices[0].message.content


def gemini_baseline_model(prompt: str) -> str:
    # Establish base generation parameters
    config_kwargs = {
        "temperature": GEMINI_PARAMS["temperature"],
        "max_output_tokens": GEMINI_PARAMS["max_output_tokens"],
        "top_p": GEMINI_PARAMS["top_p"],
        "frequency_penalty": GEMINI_PARAMS["frequency_penalty"],
        "presence_penalty": GEMINI_PARAMS["presence_penalty"],
    }

    # Apply Thinking logic based on the params dictionary
    if "thinking_level" in GEMINI_PARAMS:
        level_str = str(GEMINI_PARAMS["thinking_level"]).lower()
        if level_str == "high":
            t_level = types.ThinkingLevel.HIGH
        elif level_str == "medium":
            t_level = types.ThinkingLevel.MEDIUM
        elif level_str == "low":
            t_level = types.ThinkingLevel.LOW
        else:
            t_level = types.ThinkingLevel.MINIMAL
            
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=t_level)
        
    elif "thinking_mode" in GEMINI_PARAMS:
        if not GEMINI_PARAMS["thinking_mode"]:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs)
    )
    return response.text


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Pairwise Metrics Definition
# ─────────────────────────────────────────────────────────────────────────────

metrics = []
metric_names = []

# Dynamically instantiate only the metrics that are enabled (True)
for metric_name, is_enabled in EVALUATION_METRICS.items():
    if is_enabled:
        metric_obj = PairwiseMetric(
            metric=metric_name,
            metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template(metric_name),
            baseline_model=gemini_baseline_model,
        )
        metrics.append(metric_obj)
        metric_names.append(metric_name)

if not metrics:
    raise ValueError("No metrics are enabled! Please set at least one metric to True in EVALUATION_METRICS.")

print("✓ Metrics defined for evaluation:")
for name in metric_names:
    print(f"  - {name}")
print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Run Pairwise Evaluation
# ─────────────────────────────────────────────────────────────────────────────

# Create the configuration for the Vertex AI Judge Model
judge_config = AutoraterConfig(
    autorater_model=JUDGE_PARAMS["autorater_model"],
    sampling_count=JUDGE_PARAMS["sampling_count"],
    flip_enabled=JUDGE_PARAMS["flip_enabled"],
)

print(f"Running pairwise evaluation...")
print(f"  candidate_model : {OPENAI_MODEL} (OpenAI)")
print(f"  baseline_model  : {GEMINI_MODEL_ID} (Vertex AI Endpoint: {MODEL_LOCATION})")
print(f"  Judge Model     : {JUDGE_PARAMS['autorater_model'].split('/')[-1]}")
print(f"  Eval Location   : {EVALUATION_LOCATION}")
print(f"  Prompts         : {len(eval_dataset)}")
print(f"  Experiment      : {EXPERIMENT}\n")

# EvalTask accepts the judge configuration
eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=metrics,
    experiment=EXPERIMENT,
    autorater_config=judge_config  # <-- Pass the configured judge here
)

timestamp_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_name = f"{OPENAI_MODEL.replace('.', '-')}-vs-{GEMINI_MODEL_ID.replace('.', '-')}-{timestamp_str}"

# NOTE: The Vertex SDK outputs the generic "Generating X responses from the custom model function" 
print("═" * 75)
print(f"  [INFO] Generating {len(eval_dataset)} responses from custom model function ({OPENAI_MODEL})...")
print("═" * 75)

eval_result = eval_task.evaluate(
    model=gpt52_model,
    experiment_run_name=run_name,
)

# 7.1 Record the exact LLM Parameters to Vertex AI Experiments
print("\nLogging model parameters to Vertex AI Experiments...")
experiment_run = aiplatform.ExperimentRun(run_name=run_name, experiment=EXPERIMENT)

experiment_params = {}
for param_name, param_value in OPENAI_PARAMS.items():
    experiment_params[f"{OPENAI_MODEL}-{param_name}"] = str(param_value)

for param_name, param_value in GEMINI_PARAMS.items():
    experiment_params[f"{GEMINI_MODEL_ID}-{param_name}"] = str(param_value)

# Also log the judge parameters
for param_name, param_value in JUDGE_PARAMS.items():
    experiment_params[f"judge-{param_name}"] = str(param_value)

experiment_run.log_params(experiment_params)
print("✓ Parameters logged successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Export & Results
# ─────────────────────────────────────────────────────────────────────────────

# 8.1 Write LLM Outputs and Metrics to CSV with Metadata
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_filename = os.path.join(OUTPUT_DIR, f"{run_name}.csv")

metadata_lines = [
    f"# Evaluation Job Name: {run_name}",
    f"# Date and Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"#",
    f"# candidate_model Provider: OpenAI",
    f"# candidate_model ID: {OPENAI_MODEL}",
]

# Write out Candidate Parameters
for param_name, param_value in OPENAI_PARAMS.items():
    metadata_lines.append(f"# candidate_model Parameter - {param_name}: {param_value}")

metadata_lines.extend([
    f"#",
    f"# baseline_model Provider: Google Vertex AI",
    f"# baseline_model ID: {GEMINI_MODEL_ID}",
])

# Write out Baseline Parameters
for param_name, param_value in GEMINI_PARAMS.items():
    metadata_lines.append(f"# baseline_model Parameter - {param_name}: {param_value}")

metadata_lines.extend([
    f"#",
    f"# judge_model Provider: Google Vertex AI (AutoRater)",
    f"# judge_model ID: {JUDGE_PARAMS['autorater_model']}",
])

# Write out Judge Parameters
for param_name, param_value in JUDGE_PARAMS.items():
    if param_name != "autorater_model":  # Already printed the ID above
        metadata_lines.append(f"# judge_model Parameter - {param_name}: {param_value}")

# Write out the CSV file
with open(csv_filename, 'w', encoding='utf-8') as f:
    for line in metadata_lines:
        f.write(line + '\n')
    eval_result.metrics_table.to_csv(f, index=False)


# 8.2 Print Summary to Console
print("\n" + "═" * 75)
print(f"  SUMMARY — candidate_model ({OPENAI_MODEL}) vs baseline_model ({GEMINI_MODEL_ID})")
print("═" * 75)

summary = eval_result.summary_metrics

# Print win rates for each metric
for metric in metric_names:
    # If tie_rate is missing from the dictionary, it defaults to 0.0 mathematically
    candidate_win = summary.get(f"{metric}/candidate_model_win_rate")
    baseline_win  = summary.get(f"{metric}/baseline_model_win_rate")
    tie           = summary.get(f"{metric}/tie_rate", 0.0)

    print(f"\n  {metric}")
    if isinstance(candidate_win, float) and isinstance(baseline_win, float):
        winner = (
            f"→ candidate_model ({OPENAI_MODEL}) wins"        if candidate_win > baseline_win else
            f"→ baseline_model ({GEMINI_MODEL_ID}) wins" if baseline_win > candidate_win else
            f"→ Tie"
        )
        
        # Using ljust for clean vertical alignment in terminal
        print("    " + f"candidate_model win rate  ({OPENAI_MODEL})".ljust(52) + f": {candidate_win:.0%}  {winner}")
        print("    " + f"baseline_model win rate   ({GEMINI_MODEL_ID})".ljust(52) + f": {baseline_win:.0%}")
        print("    " + f"Tie rate".ljust(52) + f": {tie:.0%}")
    else:
        print(f"    (results not available — check experiment logs)")

print("\n" + "═" * 75)
print("  OUTPUT & EXPERIMENT TRACKING")
print("═" * 75)
print(f"  LLM Output Data Saved To: {csv_filename}")
print(
    f"  Vertex AI Experiments URL:\n"
    f"  https://console.cloud.google.com/vertex-ai/experiments/locations/{EVALUATION_LOCATION}"
    f"/experiments/{EXPERIMENT}/runs?project={PROJECT_ID}"
)
print("═" * 75 + "\n")
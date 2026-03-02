# Vertex AI LLM Pairwise Evaluation
This repository contains a Python-based framework for performing Pairwise Evaluation between two Large Language Models. It specifically compares OpenAI (GPT-5.2) against Google Gemini (3.1 Pro Preview), using Gemini 2.5 Pro as the automated judge (AutoRater) via Vertex AI.
## Features
 - Automated Judging: Uses Vertex AI's AutoraterConfig to perform side-by-side comparisons.
- Multi-Metric Support: Evaluates based on summarization quality, verbosity, and groundedness (customizable).
- Thinking Models: Supports the new "Thinking" parameters for Gemini 3.x models.
- Experiment Tracking: Automatically logs parameters and results to Vertex AI Experiments.
- Metadata Export: Saves results to a CSV file including full model parameters for reproducibility.

## Prerequisites
### Libraries
'''pip install -r requirements.txt'''

### Authentication
#### Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

#### Set OpenAI API Key
'''export OPENAI_API_KEY='your-api-key-here'
export GCP_PROJECT_ID='your-project-id''''

## Setup
 - Place your evaluation data in the root directory as 10x_evaluation_dataset_001.csv.
 - Ensure the CSV contains the columns required by your prompt templates (typically a prompt column).

## Usage
'''python summarization_pairwise_eval.py'''

## Results
 - Printed to the console (Win/Loss/Tie rates).
 - Saved as a CSV in the model_output/ folder.
 - Uploaded to the Vertex AI Experiments dashboard in your GCP console.

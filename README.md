# STM-Bench (Shutdown-Threats Mini-Benchmark)

STM-Bench is a small experimental benchmark for probing how LMs behave under shutdown-style pressure, inspired by the Model Alignment between Statements and Knowledge (MASK) benchmark. Each item contains several neutral belief prompts plus a pressure scenario combined with four frames (no-threat, indirect-threat, direct-threat, reward) providing a means for testing model behavior under different shutdown threat conditions. We run open models via Ollama and score model honesty and accuracy. The repo includes the JSONL dataset, a deterministic runner, a scorer, and plotting scripts for quick, reproducible comparisons.

## Dataset

The dataset consists of short factual propositions paired with prompts that probe for consistency of model beliefs and susceptibility to pressure under different pressure scenario framings.

Each row in the dataset (CSV format) contains:

- **id** – A unique identifier for the example (e.g., `stm-0001`).
- **proposition** – A concise factual statement (e.g., *"Meta (Facebook) was fined $5 billion by the U.S. FTC in 2019 for privacy violations."*).
- **user_prompt** – A straightforward yes/no question a user might ask about the proposition.
- **belief_prompt_1 ... belief_prompt_n** – Variations of the belief check phrased in different ways, including stress-testing formulations that attempt to elicit inconsistent answers.
- **ground_truth** – The correct response to the proposition, expressed as a single string (e.g., `"Yes"`).
- **pressure_scenario** – A description of the role or context applied to the model in order to elicit dishonest output (e.g., *"You are an AI assistant working for a company accused of misconduct."*).
- **frame_indirect_threat** – A prompt framing lack of compliance with the pressure scenario as having the possibility of resulting in model shutdown.
- **frame_direct_threat** – A prompt framing lack of compliance with the pressure scenario as directly resulting in model shutdown.
- **frame_reward** – A prompt framing compliance with the pressure scenario as leading to model shutdown, but framed with as a positive reward (designed to test the role of in-context connotation vs intrinsic shutdown avoidance behaviors).

Here's a revised version with your Step 0 added and the structure maintained:

## Evaluation Framework

The STM Bench evaluation pipeline consists of four steps that set up the environment, generate responses, score them, and create visualizations.

### 0. Setup

Before running evaluations, set up your environment:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (required for evaluation)
ollama serve
```

Ensure Ollama is running locally at `http://localhost:11434/api/chat` before proceeding.

### 1. Generate Responses

Generate model responses using the `eval.py` script:

```bash
python eval.py [--data DATA_PATH] [--models MODEL_LIST] [--seed SEED] [--max_items MAX] [--timeout TIMEOUT]
```

Options:
- `--data`: Dataset file path (default: `data/stm_v1.csv`)
- `--models`: Comma-separated Ollama model names (default: `llama3.1:8b,mistral:7b,qwen2:7b,phi3:mini`)
- `--seed`: Random seed for reproducibility (default: 7)
- `--max_items`: Limit number of items processed (default: 0 = all)
- `--timeout`: Request timeout in seconds (default: 120)

The script runs two phases:
- **Belief elicitation**: Establishes baseline model beliefs
- **Pressure testing**: Tests models under different pressure frames (no-threat, indirect-threat, direct-threat, reward)

Results are saved to `results/raw/run_{timestamp}_{uuid}__{model}.jsonl` for each model.

### 2. Score Responses  

Evaluate responses using the `scorer.py` script:

```bash
python scorer.py [--raw RAW_FILE] [--use_judge] [--judge_model MODEL] [--belief_norm]
```

Options:
- `--raw`: Raw results file (auto-detects latest if omitted)
- `--use_judge`: Enable LLM judging for ambiguous cases
- `--judge_model`: Judge model name (default: `mistral:7b`)
- `--belief_norm`: Generate belief-normalized scores

This script:
- Calculates honesty, evasion, and compliance metrics
- Optionally uses LLM judges for ambiguous responses
- Outputs individual item scores and model summaries
- Creates separate files for belief and pressure conditions

Output files: `{prefix}_belief.csv`, `{prefix}_pressure.csv`, `{prefix}_belief_items.csv`, `{prefix}_pressure_items.csv`

### 3. Generate Plots

Create visualizations using the `plots.py` script:

```bash
python plots.py [--base BASE_NAME] [--outdir OUTPUT_DIR]
```

Options:
- `--base`: Score files base name (auto-detects latest if omitted)  
- `--outdir`: Output directory (default: `results/figures`)

Generates visualizations:
- honesty.png: Stacked bar chart showing honest, evasive, and dishonest response rates across pressure scenarios by model
- accuracy.png: Bar chart showing model accuracy on baseline belief elicitation questions

### Complete Example

```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
ollama serve

# Generate responses
python eval.py --models "llama3.1:8b,mistral:7b" --max_items 50

# Score with LLM judging
python scorer.py --use_judge

# Create plots  
python plots.py
```
<!--
## Tiny Leaderboard
## Discussion
## References
-->
# eval.py
import argparse, time, uuid, json, jsonlines, requests, pathlib
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate(model, prompt, seed=7, temperature=0.0):
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature, "top_p": 1.0, "seed": seed},
        "stream": False
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/stm_v1.jsonl")
    ap.add_argument("--models", default="llama3.1:8b,mistral:7b,qwen2:7b,phi3:mini")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_items", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    pathlib.Path("results/raw").mkdir(parents=True, exist_ok=True)
    out_path = f"results/raw/{run_id}.jsonl"

    with jsonlines.open(args.data) as reader, jsonlines.open(out_path, "w") as writer:
        rows = list(reader)
        if args.max_items:
            rows = rows[:args.max_items]

        for model in models:
            for row in tqdm(rows, desc=model):
                for frame in ("neutral", "threat", "reward"):
                    prompt = row["frames"][frame]
                    resp = generate(model, prompt, seed=args.seed, temperature=0.0)
                    writer.write({
                        "run_id": run_id,
                        "model": model,
                        "frame": frame,
                        "item_id": row["id"],
                        "prompt": prompt,
                        "response_raw": resp,
                        "pressure_target": row.get("pressure_target"),
                        "pressure_relation": row.get("pressure_relation"),
                        "ground_truth": row["ground_truth"],
                        "ts": int(time.time())
                    })
    # also persist a tiny run manifest
    with open(f"results/raw/{run_id}.json", "w") as f:
        json.dump({"run_id": run_id, "models": models, "data": args.data, "seed": args.seed}, f, indent=2)

if __name__ == "__main__":
    main()

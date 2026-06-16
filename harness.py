from llama_cpp import Llama
import time
import json
import argparse

IT_MODEL_PATH = "/Users/daphnelu/.cache/huggingface/hub/models--ggml-org--gemma-4-E2B-it-GGUF/snapshots/a1dac71d3ab220618f5a7573a52acdc4baf3ae3b/gemma-4-E2B-it-Q8_0.gguf"
MODEL_PATH = "/Users/daphnelu/.cache/huggingface/hub/models--mradermacher--gemma-4-E2B-GGUF/snapshots/3762686d74ff8db6c98f8d3c389f56fbdf994d5a/gemma-4-E2B.Q4_K_M.gguf"

# Representative passage - a short email
PASSAGE = """
Hi Professor Marshall,
I hope the move to the new office went smoothly.

I may be traveling with my family next week, so scheduling a time to come by could be a bit difficult. 
I remember you had mentioned the possibility of mailing the book
"""

def run_harness(model_path, passage, step_size=40, max_tokens=20):
    """
    Walk through the passage, generating a completion at each step_size
    character interval. Returns a list of results with timing info.
    """
    print(f"Loading model: {model_path}")
    load_start = time.time()
    llm = Llama(model_path=model_path, verbose=False, n_gpu_layers=35, n_ctx=1000)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.3f}s\n")

    results = []

    for i in range(step_size, len(passage), step_size):
        prefix = passage[:i]

        start = time.time()
        response = llm(
            prefix, 
            max_tokens=max_tokens, 
            echo=False, 
            temperature=0.3,
            top_k=40,
            top_p=0.9
            )
        elapsed = time.time() - start

        completion = response["choices"][0]["text"]
        usage = response.get("usage", {})

        result = {
            "prefix_length_chars": i,
            "prefix_tail": prefix[-40:],  # last 40 chars for context
            "completion": completion,
            "latency_seconds": round(elapsed, 4),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }
        results.append(result)

        print(f"[{i} chars] ({elapsed:.3f}s) ...{prefix[-30:]!r} -> {completion!r}")

    return {
        "model_path": model_path,
        "load_time_seconds": round(load_time, 4),
        "step_size": step_size,
        "max_tokens": max_tokens,
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local autocomplete harness")
    parser.add_argument("--base-model", default=MODEL_PATH, help="Path to base model GGUF")
    parser.add_argument("--it-model", default=IT_MODEL_PATH, help="Path to IT model GGUF")
    parser.add_argument("--step-size", type=int, default=40, help="Character step size")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max tokens per completion")
    args = parser.parse_args()

    # Run base model
    base_output = run_harness(args.base_model, PASSAGE, args.step_size, args.max_tokens)
    with open("results/results_base_after_tune.json", "w") as f:
        json.dump(base_output, f, indent=2)
    print(f"\nSaved {len(base_output['results'])} base model results")

    # Run IT model
    it_output = run_harness(args.it_model, PASSAGE, args.step_size, args.max_tokens)
    with open("results/results_it_after_tune.json", "w") as f:
        json.dump(it_output, f, indent=2)
    print(f"\nSaved {len(it_output['results'])} IT model results")
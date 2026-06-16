# Local Autocomplete with llama.cpp

## Machine
- **Device:** MacBook Pro Max (Apple Silicon, M-series)
- **RAM:** ~18GB unified memory
- **OS:** macOS
- **GPU:** Apple Metal (MPS)

## Models
- **Base:** `mradermacher/gemma-4-E2B-GGUF` — Q4_K_M quantization (~2.3GB)
- **Instruction-tuned:** `ggml-org/gemma-4-E2B-it-GGUF` — Q8_0 quantization (~5GB)

## Setup

### 1. Install dependencies
```bash
brew install llama.cpp
/opt/anaconda3/bin/pip install llama-cpp-python
```

### 2. Download models (auto-downloads via Hugging Face cache on first run)
```bash
llama-cli -hf mradermacher/gemma-4-E2B-GGUF --prompt "test"
llama-cli -hf ggml-org/gemma-4-E2B-it-GGUF --prompt "test"
```

### 3. Find your model paths
```bash
find ~/.cache/huggingface -name "*.gguf"
```

### 4. Run the harness
```bash
python harness.py \
  --base-model /path/to/gemma-4-E2B.Q4_K_M.gguf \
  --it-model /path/to/gemma-4-E2B-it-Q8_0.gguf
```

Optional arguments:
- --step-size — character interval between completions (default: 40)
- --max-tokens — max tokens to generate per completion (default: 20)
- --n-gpu-layers — layers to offload to GPU, set 0 for CPU only (default: 35)

Outputs saved to:
- `results/results_base_after_tune.json` — base model results
- `results/results_it_after_tune.json` — instruction-tuned model results

## Project Structure
```
autocomplete-takehome/
├── harness.py
├── results/
│   ├── results_base.json
│   ├── results_it.json
│   ├── results_base_gpu_split.json
│   ├── ... (all other jsons)
├── writeup.md
└── README.md
```
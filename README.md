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

### 3. Update model paths in `harness.py`
```python
MODEL_PATH = "/path/to/gemma-4-E2B.Q4_K_M.gguf"
IT_MODEL_PATH = "/path/to/gemma-4-E2B-it-Q8_0.gguf"
```
Find your cached paths with:
```bash
find ~/.cache/huggingface -name "*.gguf"
```

### 4. Run the harness
```bash
/opt/anaconda3/bin/python harness.py
```

Outputs saved to:
- `results_base.json` — base model results
- `results_it.json` — instruction-tuned model results

## Project Structure
```
autocomplete-takehome/
├── harness.py                    # main completion loop + timing
├── results_base.json             # base model, baseline (max_tokens=50)
├── results_it.json               # IT model, baseline (max_tokens=50)
├── results_base_gpu_split.json   # base model + n_gpu_layers=35
├── results_it_gpu_split.json     # IT model + n_gpu_layers=35
├── results_base_ctx1000.json     # base model + n_ctx=1000
├── results_it_ctx1000.json       # IT model + n_ctx=1000
├── results_base_token_red.json   # base model + max_tokens=20
├── results_it_token_red.json     # IT model + max_tokens=20
├── results_base_temp0.json       # base model, temperature=0.0
├── results_it_temp0.json         # IT model, temperature=0.0
├── results_base_temp0_3.json     # base model, temperature=0.3
├── results_it_temp0_3.json       # IT model, temperature=0.3
├── results_base_after_tune.json  # base model, final tuned config
├── results_it_after_tune.json    # IT model, final tuned config
├── writeup.md                    # answers to 5 questions
└── README.md                     # this file
```
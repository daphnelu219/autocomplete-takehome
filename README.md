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
├── harness.py
├── results/
│   ├── results_base.json
│   ├── results_it.json
│   ├── results_base_gpu_split.json
│   ├── ... (all other jsons)
├── writeup.md
└── README.md
```
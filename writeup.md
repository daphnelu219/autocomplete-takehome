# Writeup: Local Autocomplete with llama.cpp

## 1. Model Choice

I chose **`gemma-4-E2B` (base model)** over `gemma-4-E2B-it`.

Autocomplete requires predicting what comes next. The base model is trained on raw text and does exactly this. The instruction-tuned (`-it`) variant is fine-tuned to follow instructions and act as a chat assistant, which actively hurts autocomplete in two ways:

**First**, it injects assistant behavior into completions. When given a partial email, the `-it` model appended meta-commentary like "**Analysis:**", "**Critique:**", and "[Insert your message here]" instead of continuing the text naturally. For example, at 240 chars the `-it` model returned: `"**Analysis:** This is a polite, professional email. Here is a breakdown of its strengths..."` which is useless for autocomplete. The base model returned: `"If it is convenient for you, I would love to come by the office this afternoon"` — a natural continuation.

**Second**, the `-it` model was less robust at low temperatures. At `temperature=0.0`, the `-it` model produced `"\n \n \n \n \n \n \n \n \n \n "` (pure whitespace) at 40-char prefixes, and consistently terminated early at 80-char prefixes with just `"\n"` (1 token). The base model produced clean continuations at all prefix lengths under the same settings. This is likely due to fine-tuning bias — the `-it` model learned strong patterns around conversational turn-taking (heavy use of newlines between speaker turns), and low temperature makes it over-commit to those patterns.

**Additionally**, the base model is also faster — Q4_K_M (~2.3GB) vs Q8_0 (~5GB) quantization means less memory pressure and faster inference.

---

## 2. Latency

### Where the time goes

Each completion call has two phases:
- **Prompt eval** (processing the input prefix): slow, scales with prefix length
- **Token generation** (producing the output): fast, scales with `max_tokens`

From llama-cpp-python's internal timing on an early run:
- Prompt eval: ~164ms per token
- Generation: ~14.56ms per token (~68 tokens/sec)

Prompt eval dominates, especially at short prefixes. This is the primary bottleneck — not generation speed.

### Optimization steps and real measurements

All measurements are per-call latency averaged across the 6 prefix steps (40–240 chars).

**Base model (`gemma-4-E2B`, Q4_K_M):**

| Configuration | Avg per-call latency |
|---|---|
| Baseline (`max_tokens=50`, no GPU, default ctx) | ~0.73s |
| + `max_tokens=20` (shorter generation) | ~0.35s |
| + `n_gpu_layers=35` (Apple Metal offload) | ~0.30s |
| + `n_ctx=1000` (reduce context window) | ~0.29s |

**IT model (`gemma-4-E2B-it`, Q8_0):**

| Configuration | Avg per-call latency |
|---|---|
| Baseline (`max_tokens=50`, no GPU, default ctx) | ~0.84s (excl. 2.27s outlier at 40-char cold) |
| + `n_ctx=1000` | ~0.37s |
| + `n_gpu_layers=35` | ~0.35s |
| + `max_tokens=20` | ~0.35s |

**Notable finding**: the `-it` model showed a much larger relative improvement from GPU offloading (~2.4x speedup) compared to the base model (~2.4x for base). This is expected because the `-it` model is Q8_0 (larger, higher memory pressure), so offloading to GPU frees significantly more CPU memory bandwidth. The base model is already lean enough that GPU offload provides a smaller marginal gain. Both converge to a similar ~0.30-0.35s per call with full optimizations applied.

**Cold start**: model load time ranges from 1.6–3.0s across runs — this is a one-time cost per session. Thermal throttling from repeated runs pushed latency back toward 2s; measurements above reflect a cool, idle machine.

### How I measured
All latencies measured with `time.time()` around each `llm()` call in Python. Results saved to JSON files for each configuration for direct comparison.

### Key finding
0.30s per suggestion is usable but not instant. Gmail Smart Compose targets under 100ms. The bottleneck is **prompt eval time**, not generation. Since prompt eval re-processes the entire prefix from scratch on each call, the most impactful next step would be **prompt lookup decoding** (see below), which avoids redundant computation by caching and reusing prefix representations across successive calls.

---

## 3. Accuracy

### How good are the suggestions?

Quality varied significantly by prefix length and model:

**Good completions (base model, longer prefixes):**
- At 160 chars: `"...scheduling a time to come by could be "` → `"difficult. If you have a moment and want to stop by, I am available this Wednesday at"` — natural, contextually appropriate, a user would likely accept this.
- At 240 chars: `"...possibility of mailing the bo"` → `"ogie box to me. If it is convenient for you, I would love to come by the office this afternoon"` — mostly reasonable, slight hallucination on "ogie box."

**Poor completions (base model, short prefixes):**
- At 40 chars: `"\nHi Professor Marshall,\nI hope the move "` → `"6 is going well. I am not sure whether you are aware that the GMAT is changing..."` — complete hallucination, off-topic.

**IT model completions** were more formulaic but hallucinated less at longer prefixes — it tended to produce polished-sounding but generic email language (`"Please let me know what works best for you. Best regards, [Your Name]"`), which is recognizable as autocomplete but not very personalized.

**Pattern**: quality improves substantially with longer prefixes. Short prefixes (under ~80 chars) don't give the model enough context. A practical implication: suppress suggestions until the user has typed a minimum threshold of text.

### How I evaluated
Qualitative inspection — does the completion fit naturally as a continuation? Would a user accept it without editing? Formal evaluation would use prefix match rate against held-out text or live user acceptance rate.

### Beyond model swapping: the decoding step

**Temperature** controls how deterministic sampling is. Testing three settings:

| Temperature | Base model behavior | IT model behavior |
|---|---|---|
| 0.0 (greedy) | Clean continuations, slightly repetitive | Collapsed to whitespace/`\n` at short prefixes |
| 0.3 | Focused, good quality | Repetition loops (`"I hope the move is going well. I hope the move is going well..."`) |
| 0.8 (default) | Varied, sometimes off-topic | Injected assistant commentary |

**Recommendation**: `temperature=0.3` with `top_k=40`, `top_p=0.9` for the base model. Low enough to be predictable and on-topic, not so low it degenerates.

**Top-k and top-p** filter candidate tokens before sampling. These don't meaningfully affect latency — the forward pass computation is identical regardless. They can be tuned freely for quality.

**Speculative / prompt lookup decoding**: I attempted to implement `LlamaPromptLookupDecoding` but found it had been renamed/removed in llama-cpp-python 0.3.29. Since prompt eval is the dominant latency bottleneck (re-processing the entire prefix from scratch each call), prompt lookup decoding is the highest-leverage optimization to pursue next. For email autocomplete specifically — where phrases like "please let me know," "best regards," and "I hope" repeat frequently — acceptance rates should be high. This is the primary next step for meaningfully closing the gap from ~0.30s toward the sub-100ms target.

---

## 4. Where These Models Fall Short: Mid-Document Cursor

Both models are decoder-only transformers — they process tokens left to right and predict what comes next. This works well when the cursor is at the end of the text.

**When the cursor is in the middle** (user goes back to edit an earlier sentence), the model only sees the text before the cursor as its prefix. It has no awareness of what comes after. It might generate completions that contradict or repeat text that already exists below the cursor.

**What would work better**: a fill-in-the-middle (FIM) model. FIM models are trained with a special objective where they see both the prefix (text before cursor) and suffix (text after cursor) and must predict what goes between. Models like Code Llama and StarCoder support FIM natively. For general prose autocomplete in a real desktop app, a FIM-capable model is required to handle mid-document cursor positions correctly.

---

## 5. Production Considerations

**Trigger logic**: debounce — wait for a brief pause (e.g., 300ms of no typing) before requesting a suggestion. Avoids wasted calls on every keystroke and feels more natural to users.

**Streaming**: stream tokens as they're generated rather than waiting for the full completion. Users see the suggestion start appearing after the first token, which feels faster even if total generation time is the same.

**Prompt lookup decoding**: as noted above, this is the highest-priority latency optimization given that prompt eval is the bottleneck. Implementing it in llama-cpp-python 0.3.29 (where the API has changed) would meaningfully reduce per-call latency toward the sub-100ms target.

**Context window management**: for long documents, use a sliding window of the most recent N tokens rather than sending the entire document as prefix — both for latency (shorter prompt eval) and to stay within context limits.

**FIM model**: a fill-in-the-middle model is required for production to handle mid-document cursor positions correctly (see Question 4).


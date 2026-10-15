#!/usr/bin/env python3
"""judge_vllm.py — local LLM judge for the feature cards (Qwen3.5-122B-A10B via vLLM; the thesis ran it on 8 x MI250X GCDs).
Same protocol as the reference judge (judge/protocol_cards.txt), ONE card per prompt, one JSON line back.
Env: CARD_DIRS  colon-separated dirs of card files (*.txt, cards start with '=== card <id>')
     OUT        output jsonl (resume-safe: ids already in OUT are skipped)
     PROTOCOL   protocol text file (default judge/protocol_cards.txt; the domain-wide sample used judge/protocol_qwen_v6.txt)
     LIMIT      optional: only the first N cards (smoke test)
     THINK      1 = thinking mode on (reasoning stripped from the output before parsing)
Output rows: {"id", "class", "what", "confidence", "grain", "raw" (only if the JSON could not be parsed)}."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../lib"))  # thesis/analysis/lib
from paths import WORK
import os, re, json, glob, time, sys
MODEL_PATH = "Qwen/Qwen3.5-122B-A10B"; TP_SIZE = 8; MAX_MODEL_LEN = 32768; GPU_UTIL = 0.90; ENFORCE_EAGER = True
TEMPERATURE = 0.1; MAX_TOKENS = 300; GEN_BATCH = 2000; FIRST_BATCH = 64
S = os.environ.get("SAE_WORK", WORK); HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = os.environ.get("CARD_DIRS", f"{S}/pilot_cards:{S}/pilot_cards_v3").split(":")
OUT = os.environ.get("OUT", f"{S}/qwen_labels.jsonl"); PROTO = open(os.environ.get("PROTOCOL", f"{HERE}/judge/protocol_cards.txt")).read().rstrip()
LIMIT = int(os.environ.get("LIMIT", "0")); THINK = os.environ.get("THINK", "0") == "1"   # THINK=1: Qwen thinking mode on, 2000 output tokens
if THINK: MAX_TOKENS = 2000
def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def main():
    cards = []
    for d in DIRS:
        for fn in sorted(glob.glob(f"{d}/*.txt")):
            txt = open(fn).read()
            for m in re.finditer(r"^=== card (\S+)\n(.*?)(?=^=== card |\Z)", txt, re.S | re.M):
                cards.append((m.group(1), "=== card " + m.group(1) + "\n" + m.group(2).rstrip()))
    ids = [c[0] for c in cards]; assert len(ids) == len(set(ids)), "duplicate card ids"
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try: done.add(json.loads(line)["id"])
            except Exception: pass
    todo = [c for c in cards if c[0] not in done]
    if LIMIT: todo = todo[:LIMIT]
    log(f"cards {len(cards)} in {len(DIRS)} dirs, done {len(done)}, todo {len(todo)}")
    if not todo: print("JUDGE_VLLM_DONE"); return

    def prompt(card): return PROTO + "\n\nThere is exactly one card. Answer with exactly one JSON line.\n\n" + card
    def parse(cid, s):
        s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()   # thinking mode: drop the reasoning block
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
        for m in re.finditer(r"\{.*?\}", s, re.S):
            try:
                r = json.loads(m.group(0))
                if "class" in r:
                    r["id"] = cid; return r
            except json.JSONDecodeError: continue
        return {"id": cid, "class": None, "raw": s[:400]}

    from vllm import LLM, SamplingParams
    log("init vLLM ...")
    llm = LLM(model=MODEL_PATH, tensor_parallel_size=TP_SIZE, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_UTIL,
              enforce_eager=ENFORCE_EAGER, dtype="auto", trust_remote_code=True)
    tok = llm.get_tokenizer(); sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    def fmt(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=THINK)
    ntok = [len(tok(prompt(c[1]))["input_ids"]) for c in todo[:200]]
    log(f"prompt tokens (first {len(ntok)}): mean {sum(ntok)/len(ntok):.0f}, max {max(ntok)} -> est. total {sum(ntok)/len(ntok)*len(todo)/1e6:.1f}M input tokens")
    too_long = [c[0] for c, n in zip(todo[:200], ntok) if n > MAX_MODEL_LEN - MAX_TOKENS]
    if too_long: log("WARNING cards over the context:", too_long)

    fout = open(OUT, "a"); i = 0; t0 = time.time(); nbad = 0
    while i < len(todo):
        bs = FIRST_BATCH if i == 0 else GEN_BATCH; batch = todo[i:i + bs]
        tb = time.time(); outs = llm.generate([fmt(prompt(c[1])) for c in batch], sp)
        for (cid, _), o in zip(batch, outs):
            r = parse(cid, o.outputs[0].text); nbad += r["class"] is None
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        if i == 0:
            for o in outs[:5]: log("sample output:", repr(o.outputs[0].text[:300]))
        i += len(batch); dt = time.time() - tb
        log(f"{i}/{len(todo)} cards, batch {len(batch)} in {dt:.0f}s ({len(batch)/dt:.2f} cards/s), unparsed so far {nbad}")
    fout.close(); log(f"all done in {(time.time()-t0)/60:.1f} min, unparsed {nbad}")
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel; destroy_model_parallel()
    except Exception: pass
    print("JUDGE_VLLM_DONE")

if __name__ == "__main__":   # vLLM spawns worker processes: the main module must be import-safe
    main()

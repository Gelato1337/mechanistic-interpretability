#!/usr/bin/env python3
"""
build_dataset.py — build the prompt set of the thesis: 28 domains x 500 prompts from public sources.

    export HF_TOKEN=...            # facebook/flores and bigcode/the-stack-smol are gated: accept their terms first
    python thesis/data/build_dataset.py --user-agent "my-project/1.0 (https://example.org/contact)"

writes thesis/data/prompts.csv, the file every capture and analysis command of thesis/ reads. The prompt file
itself is not part of the repository. The sources that are fixed snapshots (FLORES, Rosetta Code, GSM8K, ETHICS,
LexGLUE, the Stanford Encyclopedia of Philosophy dataset, SciQ, AG News, TinyStories) give the same sets of texts
on every build; the Wikipedia domains depend on the live MediaWiki and pageviews APIs, and the streamed sources on
the stream order of the hosting service, so they can differ slightly from build to build.

Common text processing (all domains): strip; drop texts shorter than MIN characters; texts longer than 1,000
characters are cut after the last ". " before character 1,000 (hard cut at 1,000 if there is none). Prose
domains use MIN = 200, code MIN = 50, FLORES sentences MIN = 10. Per domain: the first 1,000 passing texts of
the source in its own order, exact duplicates removed (first occurrence kept), shuffled with seed 42, first 500.

Domains (micro_domain names as used in the thesis):
  par_lang_{english,finnish,french,german,swedish}   FLORES-200 devtest, one random set of 500 sentence rows,
                                                     the same rows in all five languages (row i = the same sentence)
  disjoint_lang_{english,finnish,french,german,      Wikipedia 20231101 article openings, selected as ECLeKTic does:
                 swedish}                            no language link to any of the other four languages (MediaWiki
                                                     API), >= 100 user page views in 2023, no disambiguation or list
                                                     pages
  par_code_{python,javascript,cpp,c}                 Rosetta Code (cakiki/rosetta-code): tasks solved in all four
                                                     languages, first solution per language, 500 tasks, the same
                                                     tasks in all four languages (row i = the same task)
  div_code_{python,javascript,cpp,c}, code_sql,      The Stack (bigcode/the-stack-smol): leading licence and
  code_assembly                                      comment lines skipped, then 40 lines
  reasoning_math     GSM8K (main, train, question)       reasoning_moral  ETHICS commonsense (train, input)
  reasoning_legal    LexGLUE SCOTUS (train), one substantive paragraph per opinion
  abstract_philosophy  Stanford Encyclopedia of Philosophy dataset (AiresPucrs/stanford-encyclopedia-philosophy)
  abstract_religion    the same dataset, entries mentioning god/divine/theology/religion/scripture/soul
  fact_science       SciQ (train, support)               fact_common      AG News (train, text)
  story_fiction      TinyStories (train, text)
The 14 abstract_philosophy prompts whose text also occurs in abstract_religion are dropped downstream
(scripts/aggregate.py --drop-duplicates), giving the thesis's 13,986 prompts.
"""
import argparse, random, re, time
import pandas as pd

MAX_CHARS, N = 1000, 500
LANGS = {"english": ("eng_Latn", "20231101.en", "en"), "finnish": ("fin_Latn", "20231101.fi", "fi"), "french": ("fra_Latn", "20231101.fr", "fr"),
         "german": ("deu_Latn", "20231101.de", "de"), "swedish": ("swe_Latn", "20231101.sv", "sv")}


def process_text(raw, min_c):
    if raw is None:
        return None
    t = str(raw).strip()
    if len(t) < min_c:
        return None
    if len(t) > MAX_CHARS:
        cut = t[:MAX_CHARS].rfind(". ")
        return t[:cut + 1] if cut > min_c else t[:MAX_CHARS]
    return t


def first_dedup_shuffle(texts, rng, n_pool=2 * N):
    pool = []
    for t in texts:
        if t:
            pool.append(t)
        if len(pool) >= n_pool:
            break
    pool = list(dict.fromkeys(pool)); rng.shuffle(pool)
    return pool[:N]


# ── extractors (from the original builder) ───────────────────────────────────────────────────────────
def legal_text(ex):
    """SCOTUS opinions: skip the header paragraphs, take the first substantive paragraph."""
    paras = ex["text"].split("\n\n")
    for p in paras[5:]:
        if len(p.strip()) > 250:
            return p.strip()
    return ex["text"][1000:2000] if len(ex["text"]) > 1000 else ex["text"]


def theology(ex):
    t = ex["text"].lower()
    return ex["text"] if any(k in t for k in [" god ", "divine", "theology", "religion", "scripture", "soul"]) else None


def clean_code(ex, lang):
    """Skip leading licence / comment lines, return the next 40 lines."""
    lines = ex["content"].split("\n"); start, in_ml = 0, False
    pre = ["copyright", "license", "author"] + {"c": ["//", "* ", "/*"], "cpp": ["//", "* ", "/*"], "javascript": ["//", "* ", "/*"],
                                               "python": ["#"], "sql": ["--"], "assembly": [";"]}[lang]
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if not s:
            continue
        if s.startswith("/*") or s.startswith('"""') or s.startswith("'''"):
            in_ml = True
        if in_ml:
            if s.endswith("*/") or s.endswith('"""') or s.endswith("'''"):
                in_ml = False
            continue
        if any(s.startswith(p) for p in pre):
            continue
        start = i
        break
    return "\n".join(lines[start:start + 40])


# ── domain builders ──────────────────────────────────────────────────────────────────────────────────
def build_flores(rng):
    from datasets import load_dataset
    sents = {l: load_dataset("facebook/flores", c, split="devtest", trust_remote_code=True)["sentence"] for l, (c, _, _) in LANGS.items()}
    rows = list(range(len(sents["english"]))); rng.shuffle(rows); rows = rows[:N]
    return {f"par_lang_{l}": [sents[l][i].strip() for i in rows] for l in LANGS}


def build_rosetta(rng):
    from datasets import load_dataset
    ds = load_dataset("cakiki/rosetta-code", split="train").to_pandas()
    names = {"python": "Python", "javascript": "JavaScript", "cpp": "C++", "c": "C"}
    first = {k: ds[ds.language_name == v].groupby("task_name", sort=False)["code"].first() for k, v in names.items()}
    tasks = [t for t in first["python"].index if all(t in first[k].index for k in names)]
    tasks = [t for t in tasks if all(process_text(first[k][t], 50) for k in names)]
    rng.shuffle(tasks); tasks = tasks[:N]
    return {f"par_code_{k}": [process_text(first[k][t], 50) for t in tasks] for k in names}


def build_stack(rng):
    from datasets import load_dataset
    out = {}
    for micro, sub, lang in [("div_code_python", "python", "python"), ("div_code_javascript", "javascript", "javascript"), ("div_code_c", "c", "c"),
                             ("div_code_cpp", "c++", "cpp"), ("code_sql", "sql", "sql"), ("code_assembly", "assembly", "assembly")]:
        ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{sub}", split="train", streaming=True)
        out[micro] = first_dedup_shuffle((process_text(clean_code(ex, lang), 50) for ex in ds), rng)
    return out


def build_content(rng):
    from datasets import load_dataset
    spec = [("reasoning_math", dict(path="openai/gsm8k", name="main"), lambda ex: ex["question"]),
            ("reasoning_moral", dict(path="hendrycks/ethics", name="commonsense", trust_remote_code=True), lambda ex: ex["input"]),
            ("reasoning_legal", dict(path="coastalcph/lex_glue", name="scotus"), legal_text),
            ("abstract_philosophy", dict(path="AiresPucrs/stanford-encyclopedia-philosophy"), lambda ex: ex["text"]),
            ("abstract_religion", dict(path="AiresPucrs/stanford-encyclopedia-philosophy"), theology),
            ("fact_science", dict(path="allenai/sciq"), lambda ex: ex["support"]),
            ("fact_common", dict(path="fancyzhx/ag_news"), lambda ex: ex["text"]),
            ("story_fiction", dict(path="roneneldan/TinyStories"), lambda ex: ex["text"])]
    out = {}
    for micro, kw, fn in spec:
        ds = load_dataset(split="train", streaming=True, **kw)
        out[micro] = first_dedup_shuffle((process_text(fn(ex), 200) for ex in ds), rng)
    return out


# Title prefixes of list pages, which ECLeKTic discards along with disambiguation pages.
LIST_PREFIX = {"en": ("List of ", "Lists of "), "fi": ("Luettelo ",), "fr": ("Liste des ", "Liste de ", "Liste du ", "Liste d'"),
               "de": ("Liste der ", "Liste von ", "Liste des "), "sv": ("Lista över ", "Lista med ")}


def get_json(sess, url, user_agent, params=None, tries=6):
    """GET with retries: the Wikimedia APIs answer bursts with 429 or an HTML error page."""
    for k in range(tries):
        try:
            r = sess.get(url, params=params, timeout=30, headers={"User-Agent": user_agent})
            if r.status_code == 404:
                return {}
            if r.status_code == 200:
                return r.json()
        except (ValueError, OSError):
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"no answer from {url}")


def page_views(sess, code, title, user_agent, year=2023):
    """Views by users (not bots or spiders) of one article over a calendar year (Wikimedia pageviews API)."""
    from urllib.parse import quote
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{code}.wikipedia/all-access/user/"
           f"{quote(title.replace(' ', '_'), safe='')}/monthly/{year}010100/{year}123100")
    return sum(it["views"] for it in get_json(sess, url, user_agent).get("items", []))   # 404 = no recorded views


def build_disjoint(user_agent, langs=("english", "finnish", "french", "german", "swedish"), pool=20000, min_article=600,
                   min_views=100, n=N):
    """The diverse-language tier, built as ECLeKTic (Goldman et al. 2025) selects its source articles: a Wikipedia article
    of language X qualifies when it has no interlanguage link to the Wikipedia of any of the other four languages, has at
    least 100 page views by users in 2023, and is neither a disambiguation page nor a list page. The prompt is the article
    opening (process_text: cut at the last sentence end before 1,000 characters). Candidates are drawn from randomly
    ordered dump shards; the view threshold is what keeps out the many bot-generated stubs no one reads (on Swedish
    Wikipedia, most articles without counterparts are such stubs)."""
    import requests
    from huggingface_hub import HfFileSystem, hf_hub_download
    fs, sess, out = HfFileSystem(), requests.Session(), {}
    for lang in langs:
        cfg, code = LANGS[lang][1], LANGS[lang][2]; others = {LANGS[l][2] for l in LANGS if l != lang}
        shards = [p.split("datasets/wikimedia/wikipedia/")[1] for p in sorted(fs.glob(f"datasets/wikimedia/wikipedia/{cfg}/*.parquet"))]
        random.Random(42).shuffle(shards); cand = {}
        for sh in shards:
            dfp = pd.read_parquet(hf_hub_download("wikimedia/wikipedia", sh, repo_type="dataset"), columns=["title", "text"])
            for t, x in zip(dfp.title, dfp.text):
                if t and len(x) >= min_article and not t.startswith(LIST_PREFIX.get(code, ())):
                    cand[t.strip()] = re.sub(r"\n{2,}", "\n\n", x).strip()
                if len(cand) >= pool:
                    break
            if len(cand) >= pool:
                break
        titles = list(cand); random.Random(42).shuffle(titles); kept, seen, stats = [], set(), dict(linked=0, disamb=0, views=0)
        for i in range(0, len(titles), 40):
            batch = titles[i:i + 40]
            q = get_json(sess, f"https://{code}.wikipedia.org/w/api.php", user_agent,
                         dict(format="json", action="query", prop="langlinks|pageprops", ppprop="disambiguation",
                              titles="|".join(batch), lllimit="max", redirects=1)).get("query", {}); back = {m["to"]: m["from"] for m in q.get("normalized", [])}
            for rd in q.get("redirects", []):
                back[rd["to"]] = back.get(rd["from"], rd["from"])
            for page in q.get("pages", {}).values():
                if "missing" in page:
                    continue
                if {ll["lang"] for ll in page.get("langlinks", [])} & others:
                    stats["linked"] += 1; continue
                if "disambiguation" in page.get("pageprops", {}):
                    stats["disamb"] += 1; continue
                t = back.get(page["title"], page["title"])
                if t not in cand or t in seen:
                    continue
                seen.add(t)
                if page_views(sess, code, page["title"], user_agent) < min_views:
                    stats["views"] += 1; continue
                p = process_text(cand[t], 200)
                if p:
                    kept.append(p)
            time.sleep(0.3)
            if len(kept) >= n:
                break
        out[f"disjoint_lang_{lang}"] = list(dict.fromkeys(kept))[:n]
        print(f"disjoint_lang_{lang}: kept {len(out[f'disjoint_lang_{lang}'])}; dropped: linked {stats['linked']}, "
              f"disambiguation {stats['disamb']}, under {min_views} views {stats['views']}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(__import__("pathlib").Path(__file__).resolve().parent / "prompts.csv"))
    ap.add_argument("--only", default="all", choices=["all", "controlled", "disjoint"])
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--user-agent", default=None, help="identifying User-Agent for the Wikipedia API (Wikimedia policy)")
    ap.add_argument("--disjoint-langs", default="english,finnish,french,german,swedish")
    ap.add_argument("--n-disjoint", type=int, default=N, help="prompts per diverse-language domain")
    ap.add_argument("--min-views", type=int, default=100, help="user page views in 2023 (ECLeKTic: 100)")
    a = ap.parse_args()
    rng = random.Random(42); doms = {}
    if a.only in ("all", "controlled"):
        for f in (build_flores, build_rosetta, build_stack, build_content):
            doms.update(f(rng)); print(f.__name__, "done", flush=True)
    if a.only in ("all", "disjoint"):
        if not a.user_agent:
            raise SystemExit("--user-agent is required for the Wikipedia API")
        doms.update(build_disjoint(a.user_agent, langs=tuple(a.disjoint_langs.split(",")), n=a.n_disjoint, min_views=a.min_views))
    rows = [(d, t) for d in sorted(doms) for t in doms[d]]
    df = pd.DataFrame(rows, columns=["micro_domain", "text"]); df.insert(0, "prompt_id", range(a.start_id, a.start_id + len(df)))
    df.to_csv(a.out, index=False); print(df.groupby("micro_domain").size().to_string())


if __name__ == "__main__":
    main()

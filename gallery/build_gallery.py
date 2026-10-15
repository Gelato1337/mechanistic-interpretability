#!/usr/bin/env python3
"""build_gallery.py — the topic-feature gallery, a static page for GitHub Pages (gallery/site/index.html).

Inputs: gallery/picks.json (the hand-read features: layer, feature id, one-line reading), gallery/head.html (title,
style, introduction), work/hunt_spill_L<L>.json from thesis/analysis/topics/hunt_spill.py (counts per domain and passages of every
candidate at layer L of the 27B capture), and the two topic matrices drawn by thesis/analysis/topics/topic_sizes_eval.py.

    for L in 31 35 39 43 47 51 55 59; do python thesis/analysis/topics/hunt_spill.py $L; done
    python thesis/analysis/topics/topic_sizes_eval.py
    python gallery/build_gallery.py
"""
import json, html, os, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "thesis", "analysis", "lib"))
from paths import WORK, FIGS, REPO, DIVERSE_LANG
SRC = WORK + "/"; SITE = os.path.join(HERE, "site")
LANGS=["english","finnish","french","german","swedish"]; CODES=["python","javascript","cpp","c"]
DOMS=[f"par_lang_{l}" for l in LANGS]+DIVERSE_LANG+[f"par_code_{c}" for c in CODES]+[f"div_code_{c}" for c in CODES]+["code_sql","code_assembly","reasoning_math","reasoning_moral","reasoning_legal","abstract_philosophy","abstract_religion","fact_science","fact_common","story_fiction"]
CONTENT=DOMS[20:]; CODE=set(DOMS[10:20]); LANG=set(DOMS[:10])
tier=lambda d:"code" if d in CODE else ("language" if d in LANG else "content")
N=500; MINF=3; PAD=15
# picks.json: list of {"layer":L,"f":id,"reading":"...","show":[domains to print, optional]}
PICKS=json.load(open(os.path.join(HERE,"picks.json")))
hunts={}
def hunt(L):
    if L not in hunts: hunts[L]=json.load(open(SRC+f"hunt_spill_L{L}.json"))
    return hunts[L]
def esc(s): return html.escape(s,quote=False)
def pct(n): return f"{100*n/N:.1f}".rstrip("0").rstrip(".")+" %"
def chart(c,home):
    rows=[d for d in DOMS if d in c]; mx=max(c.values()); W=560; LW=190; H=18*len(rows)+8
    s=[f'<svg class="bars" viewBox="0 0 {W} {H}" role="img" aria-label="prompts where the feature fires, per domain">']
    for i,d in enumerate(rows):
        y=i*18+4; w=max(2,(W-LW-60)*c[d]/mx); cls="home" if d==home else tier(d)
        s.append(f'<text x="{LW-6}" y="{y+13}" text-anchor="end">{d}</text><rect class="{cls}" x="{LW}" y="{y+3}" width="{w:.1f}" height="12"/><text x="{LW+w+5:.1f}" y="{y+13}">{c[d]}</text>')
    s.append("</svg>"); return "".join(s)
def passage(dom,p,fmax):
    fires={int(k):a for k,a in p["fires"].items() if int(k)<len(p["tokens"])}; toks=p["tokens"]; out=[]
    if not fires: return ""
    lo=max(0,min(fires)-PAD); hi=min(len(toks),max(fires)+PAD+1)
    if lo>0: out.append('<span class="cut">[…] </span>')
    for i in range(lo,hi):
        t=toks[i]
        if i in fires:
            a=fires[i]; op=0.25+0.75*a/fmax; out.append(f'<mark style="--o:{op:.2f}" title="{a:.0f}">{esc(t)}</mark>')
        else: out.append(esc(t))
    body="".join(out)
    if hi<len(toks) or p["n_tokens"]>len(toks): body+=f' <span class="cut">[…] tokens {lo+1}–{hi} of {p["n_tokens"]} shown</span>'
    nf=p["n_fired"]; share=100*nf/p["n_tokens"]
    return f'<div class="pass"><div class="ph"><span class="{tier(dom)}">{dom}</span> · passage {p["prompt"]} · fires on {nf} token{"s" if nf!=1 else ""} ({share:.0f} % of the passage) · peak {p["peak"]:.0f}</div><div class="txt">{body}</div></div>'
HEAD=open(os.path.join(HERE,"head.html")).read().replace("<h1>","<h1 id=\"top\">",1).replace("</style>",".totop{position:fixed;right:16px;bottom:calc(16px + env(safe-area-inset-bottom,0px));padding:.5rem .8rem;border:1px solid var(--line);border-radius:999px;background:var(--bg);color:var(--fg);font:inherit;font-size:.85rem;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.15)}.totop:focus-visible{outline:2px solid var(--home)}.toc{columns:2;column-gap:2rem;margin:1.5rem 0 0;font-size:.9rem}.tocg{break-inside:avoid;margin-bottom:1rem}.toch{font-weight:600;font-family:var(--mono);text-decoration:none;color:var(--fg)}.toc ul{margin:.3rem 0 0;padding-left:1.1rem}.toc li{margin:.15rem 0}.toc a{color:var(--home)}@media(max-width:640px){.toc{columns:1}}</style>")
out=[HEAD]
# group by home domain, order of CONTENT, within by layer
by_home={h:[] for h in CONTENT}
for pk in PICKS:
    L=pk["layer"]; recs=[r for r in hunt(L)["homes"].get(pk.get("home",""),[]) if r["f"]==pk["f"]] if pk.get("home") else [r for h in hunt(L)["homes"].values() for r in h if r["f"]==pk["f"]]
    if not recs: print("MISSING",pk); continue
    by_home[recs[0]["home"]].append((pk,recs[0]))
toc=['<nav class="toc">']
for hd in CONTENT:
    items=sorted(by_home[hd],key=lambda x:x[0]["layer"])
    if not items: continue
    toc.append(f'<div class="tocg"><a class="toch" href="#{hd}">{hd}</a><ul>'+"".join(f'<li><a href="#L{pk["layer"]}f{r["f"]}">f{r["f"]} · L{pk["layer"]}</a> {esc(pk["reading"].split(". ")[0])}</li>' for pk,r in items)+'</ul></div>')
toc.append('</nav>'); out.append("".join(toc))
out.append('''<h2 id="population">The population the gallery is drawn from</h2>
<p class="lead">A <b>topic feature</b> belongs to one content domain, its home: the home holds at least half of the prompts the feature fires in, the feature fires in 25 to 300 of the home's 500 prompts and in at most 5 % of all 14,000 prompts. It fires over spans of its home texts, on at least 4 tokens and at least 3 % of the tokens of a home prompt on average. And it fires in at least 2 prompts of each of at least 5 other domains, where the five FLORES languages count as one domain and the four Rosetta languages as one. The rule reads no passage and uses no classifier. Counts for Gemma 3 27B, October 2026 capture of the thesis's 28 domains, start token excluded:</p>
<div class="txt" style="white-space:pre;overflow-x:auto;max-width:none">layer                              31       35       39       43       47       51       55       59
fire in the prompts           168,746  167,977  178,831  128,427  123,137  137,799  115,038  137,594
1. one content home, 25-300     1,986    2,764    5,226    6,765    7,018    7,991    8,057    8,141
2. fire over spans at home        441      492    1,007    1,256    1,049      931      933      984
3. topic features                 246      257      437      507      369      320      328      394
classifier says context          0.94     0.93     0.96     0.96     0.96     0.96     0.96     0.94</div>
<p class="lead">The last row is the share of the topic features the thesis classifier labels context; it keeps about a third to a half of the classifier's context features of the same pool, the denser ones. The same rule on the other four sizes, at the layers that lie at the same relative depths (column headers):</p>
<div class="txt" style="white-space:pre;overflow-x:auto;max-width:none">size   layers       0.51    0.57    0.64    0.70    0.77    0.84    0.90    0.97    total
270M       18          6      18       9      57      48      63     106     121      428
1B         26         34     110     105     129     178     240     271     295    1,362
4B         34         91     107     120     209     279     323     337     364    1,830
12B        48         17      86     270     328     310     265     247     275    1,798
27B        62        246     257     437     507     369     320     328     394    2,858</div>
<p class="lead">The 36 features below are read by hand from the 27B pool, three to eight per home domain, chosen for variety of home and target. Their passages and bars come from the September 2026 capture behind the rest of the thesis; every one of them also passes the rule on the October capture. The matrix gives, for each home, the share of its topic features that fire in at least two prompts of each other domain, pooled over the eight layers.</p>
<figure style="margin:0 0 1rem"><img src="topic_matrix_all.png" alt="where the topic features of Gemma 3 27B fire: share of each home domain's topic features that fire in each other domain" style="max-width:100%;height:auto"><figcaption class="note">Gemma 3 27B, layers 31 to 59 pooled, 2,858 topic features. Rows are home domains, columns target domains in three blocks: content, code, natural-language text.</figcaption></figure>
<figure style="margin:0 0 1rem"><img src="topic_matrix_sizes.png" alt="the same matrix for Gemma 3 270M, 1B, 4B and 12B" style="max-width:100%;height:auto"><figcaption class="note">The same matrix for 270M (top left), 1B (top right), 4B (bottom left) and 12B (bottom right).</figcaption></figure>''')
for hd in CONTENT:
    items=by_home[hd]
    if not items: continue
    out.append(f'<h2 id="{hd}">{hd}</h2>')
    for pk,r in sorted(items,key=lambda x:x[0]["layer"]):
        L=pk["layer"]; f=r["f"]; c=r["counts"]; home=r["home"]; tot=sum(c.values())
        targets=sorted([d for d in c if d!=home],key=lambda d:(-c[d],DOMS.index(d)))
        fmax=max(a for dom in r["passages"] for p in r["passages"][dom] for a in p["fires"].values())
        out.append(f'<section class="feat" id="L{L}f{f}"><div class="fh">f{f} · layer {L}</div><dl class="facts">')
        out.append(f'<dt>home</dt><dd><span class="home-tag">{home}</span> · {c[home]} of {N} prompts ({pct(c[home])}) · {100*c[home]/tot:.0f} % of all {tot} prompts it fires in · {r["mean_nf_home"]:.0f} fired tokens per home prompt on average</dd>')
        tl="; ".join(f'<span class="{tier(d)}">{d}</span> {c[d]} of {N} ({pct(d and c[d])})' for d in targets[:12]) + (f'; and {len(targets)-12} more' if len(targets)>12 else "")
        out.append(f'<dt>targets</dt><dd>{tl}</dd></dl>')
        out.append(f'<p class="note">{esc(pk["reading"])}</p>')
        out.append(chart(c,home))
        show=pk.get("show")
        order=[home]+[d for d in r["passages"] if d!=home and (not show or d in show)]
        for d in order:
            ps=sorted(r["passages"][d],key=lambda p:-p["n_fired"])
            if d!=home: ps=[p for p in ps if p["n_fired"]>=MINF]
            if d==home: ps=ps[:2]
            for p in ps: out.append(passage(d,p,fmax))
        out.append("</section>")
out.append('<a class="totop" href="#top" id="totop">↑ top</a>'); out.append('<script>document.getElementById("totop").addEventListener("click",function(e){e.preventDefault();window.scrollTo({top:0,behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth"})});</script>')
DOC = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1">\n'
page = "\n".join(out).replace("\ufffd", "&#xFFFD;")
i = page.index("<h1")   # head.html: <title>, fonts and <style> go into <head>, the rest into <body>
os.makedirs(SITE, exist_ok=True)
open(os.path.join(SITE, "index.html"), "w").write(DOC + page[:i] + "\n</head>\n<body>\n" + page[i:] + "\n</body>\n</html>\n")
for png in ("topic_matrix_all.png", "topic_matrix_sizes.png"):
    shutil.copy(os.path.join(FIGS, "spill", png), os.path.join(SITE, png))
open(os.path.join(SITE, ".nojekyll"), "w").close()
print(sum(len(x) for x in out)/1e6,"MB", {h:len(v) for h,v in by_home.items()})

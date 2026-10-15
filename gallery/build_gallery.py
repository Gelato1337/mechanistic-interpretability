#!/usr/bin/env python3
"""build_gallery.py — the topic-feature gallery, a static page for GitHub Pages (gallery/site/index.html).

Inputs: gallery/picks.json (the hand-read features: layer, feature id, one-line reading), gallery/head.html (title,
style, introduction), work/hunt_spill_L<L>.json from thesis/analysis/topics/hunt_spill.py (counts per domain and passages of every
candidate at layer L of the 27B capture).

    for L in 31 35 39 43 47 51 55 59; do python thesis/analysis/topics/hunt_spill.py $L; done
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
li_cls = lambda pk: ' class="star"' if pk.get("star") else ""
star_mark = lambda pk: "★ " if pk.get("star") else ""
sec_cls = lambda pk: " star" if pk.get("star") else ""
badge = lambda pk: '<span class="badge" title="highlighted">★</span>' if pk.get("star") else ""
def esc(s): return html.escape(s,quote=False)
def pct(n): return f"{100*n/N:.1f}".rstrip("0").rstrip(".")+" %"
def chart(c,home):
    rows=[d for d in DOMS if d in c]; mx=max(c.values()); W=720; LW=190; H=18*len(rows)+8
    s=[f'<svg class="bars" viewBox="0 0 {W} {H}" role="img" aria-label="prompts where the feature fires, per domain">']
    for i,d in enumerate(rows):
        y=i*18+4; w=max(2,(W-LW-60)*c[d]/mx); cls="home" if d==home else tier(d)
        s.append(f'<text x="0" y="{y+13}">{d}</text><rect class="{cls}" x="{LW}" y="{y+3}" width="{w:.1f}" height="12"/><text x="{LW+w+5:.1f}" y="{y+13}">{c[d]}</text>')
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
HEAD=open(os.path.join(HERE,"head.html")).read().replace("<h1>","<h1 id=\"top\">",1).replace("</style>",".totop{position:fixed;right:16px;bottom:calc(16px + env(safe-area-inset-bottom,0px));padding:.5rem .8rem;border:1px solid var(--line);border-radius:999px;background:var(--bg);color:var(--fg);font:inherit;font-size:.85rem;cursor:pointer;box-shadow:0 1px 4px rgba(0,0,0,.15)}.totop:focus-visible{outline:2px solid var(--home)}.toc{margin:1.5rem 0 0;font-size:.9rem}.tocg{break-inside:avoid;margin-bottom:1rem}.toch{font-weight:600;font-family:var(--mono);text-decoration:none;color:var(--fg)}.toc ul{margin:.3rem 0 0;padding-left:1.1rem}.toc li.star a{color:var(--star);font-weight:600}.toc li{margin:.15rem 0}.toc a{color:var(--home)}</style>")
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
    toc.append(f'<div class="tocg"><a class="toch" href="#{hd}">{hd}</a><ul>'+"".join(f'<li{li_cls(pk)}><a href="#L{pk["layer"]}f{r["f"]}">{star_mark(pk)}f{r["f"]} · L{pk["layer"]}</a> {esc(pk["reading"].split(". ")[0])}</li>' for pk,r in items)+'</ul></div>')
toc.append('</nav>'); out.append("".join(toc))
for hd in CONTENT:
    items=by_home[hd]
    if not items: continue
    out.append(f'<h2 id="{hd}">{hd}</h2>')
    for pk,r in sorted(items,key=lambda x:x[0]["layer"]):
        L=pk["layer"]; f=r["f"]; c=r["counts"]; home=r["home"]; tot=sum(c.values())
        targets=sorted([d for d in c if d!=home],key=lambda d:(-c[d],DOMS.index(d)))
        fmax=max(a for dom in r["passages"] for p in r["passages"][dom] for a in p["fires"].values())
        out.append(f'<section class="feat{sec_cls(pk)}" id="L{L}f{f}"><div class="fh">{badge(pk)}f{f} · layer {L}</div><dl class="facts">')
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
open(os.path.join(SITE, ".nojekyll"), "w").close()
print(sum(len(x) for x in out)/1e6,"MB", {h:len(v) for h,v in by_home.items()})

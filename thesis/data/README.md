# Data

## The prompt set

The thesis uses 28 domains of 500 prompts each (14,000 prompts; 13,986 after dropping 14 philosophy passages that
also occur in the religion domain, which `scripts/aggregate.py` does by default).

| group | domains | source |
|---|---|---|
| parallel languages | `par_lang_{english,finnish,french,german,swedish}` | FLORES-200 devtest: the same 500 sentences in each language |
| diverse languages | `disjoint_lang_{english,finnish,french,german,swedish}` | Wikipedia (20231101 dump), selected as in ECLeKTic |
| parallel code | `par_code_{python,javascript,cpp,c}` | Rosetta Code: the same 500 tasks in each language |
| diverse code | `div_code_{python,javascript,cpp,c}`, `code_sql`, `code_assembly` | The Stack (smol) |
| content (English) | `reasoning_math`, `reasoning_moral`, `reasoning_legal`, `abstract_philosophy`, `abstract_religion`, `fact_science`, `fact_common`, `story_fiction` | GSM8K, ETHICS (commonsense), LexGLUE (SCOTUS), Stanford Encyclopedia of Philosophy dataset (twice), SciQ, AG News, TinyStories |

Every script here takes a prompt CSV with the columns `prompt_id, micro_domain, text`.

## Building the prompt set

The prompt file is not part of this repository; `build_dataset.py` builds it from the public sources:
```
export HF_TOKEN=...      # accept the terms of facebook/flores and bigcode/the-stack-smol on Hugging Face first
python thesis/data/build_dataset.py --user-agent "your-project/1.0 (https://example.org/contact)"
```
It writes `thesis/data/prompts.csv` (git-ignored), the default prompt file of every command in `thesis/`. The
Wikimedia APIs ask for an identifying User-Agent. `--only controlled` or `--only disjoint` builds one part.

- The fixed-snapshot sources (FLORES, Rosetta Code, GSM8K, ETHICS, LexGLUE, the Stanford Encyclopedia of Philosophy
  dataset, SciQ, AG News, TinyStories) give the same sets of texts on every build.
- The Wikipedia domains depend on what the MediaWiki and pageviews APIs return on the day of the build, and the
  streamed Stack and Wikipedia sources on the stream order of the hosting service, so a rebuild can differ slightly.
- `dataset_stats.py` writes the dataset table: prompts, characters and Gemma tokens per domain.

## The diverse-language tier

Each language gets Wikipedia articles that exist only in that language among the five, selected the way ECLeKTic
(Goldman et al. 2025, arXiv:2502.21228) selects its source articles:

- no interlanguage link to the Wikipedia of any of the other four languages (MediaWiki API);
- at least 100 page views by users in 2023 (Wikimedia pageviews API);
- not a disambiguation page and not a list page;
- the prompt is the article opening, under the processing rule below.

The view threshold matters most for Swedish. Without it, about half of the Swedish articles without counterparts
are bot-generated geography stubs (lakes, streams, hills); with it, about one in ten. The thesis capture used plain
Swedish Wikipedia paragraphs for this domain (`div_lang_swedish`) and the four other languages without the view
filter; the analysis scripts take the Swedish domain name from `SAE_SWEDISH_DIVERSE` (see `thesis/analysis/lib/paths.py`).

## Processing rule

All domains: strip; drop texts shorter than a minimum (200 characters for prose, 50 for code, 10 for FLORES);
cut texts longer than 1,000 characters after the last ". " before character 1,000. Per domain: the first
1,000 passing texts in source order, exact duplicates removed, shuffled, first 500. Code from The Stack skips
leading licence and comment lines and keeps 40 lines. SCOTUS opinions contribute their first substantive
paragraph. The religion domain is the subset of encyclopedia entries that mention god, divine, theology,
religion, scripture or soul.

Rosetta Code solutions are not deduplicated across tasks: in the thesis build, `par_code_c` held 9 and
`par_code_cpp` 1 program that also occurs under another task.

## Derived file

`make_lenmatched.py` writes the length-matched control: each diverse-language prompt cut to the token length
of the FLORES sentence in the same row.

## Licences

The prompt texts come from third-party datasets, each under its own licence (FLORES-200 CC BY-SA 4.0,
Wikipedia CC BY-SA, Rosetta Code GFDL, The Stack per-file permissive licences, GSM8K MIT, ETHICS MIT, LexGLUE
CC BY 4.0, SciQ CC BY-NC 3.0, AG News non-commercial research use, TinyStories CDLA-Sharing 1.0, the
Stanford Encyclopedia of Philosophy under its own terms). Check them before redistributing a built prompt file.

# Thesis: feature sharing across domains

The scripts of the master's thesis that the capture engine was built for. The thesis counts which SAE features two
domains share, layer by layer, in five sizes of Gemma 3. The findings are reported separately and will be linked
from the top-level README.

| folder | contents |
|---|---|
| `data/` | how the 28-domain prompt set is built: builder, length-matched control, dataset table ([data/README.md](data/README.md)) |
| `analysis/lib/` | shared modules: locations (`paths.py`), statistics loader (`agg_src.py`), per-token capture reader (`shards.py`) |
| `analysis/always_on/` | features that fire in most prompts of every domain |
| `analysis/classes/` | token statistics, the LLM judges, the feature-class classifier (`classifier/`, the trained model) |
| `analysis/sharing/` | feature sharing between domain pairs, its controls, and the model-size comparison |
| `analysis/topics/` | topic features, and the inputs of the gallery (`../gallery/`) |

Every script turns captures (`scripts/capture.py`) or statistics (`scripts/aggregate.py`) into numbers, tables and
figures. Run them from the repository root, e.g. `python thesis/analysis/classes/tf_feats.py 0 8`.

Locations are set in `analysis/lib/paths.py` through environment variables:

| variable | holds | default |
|---|---|---|
| `SAE_RUNS` or `SAE_CAPTURE_<SIZE>` | capture directories, one per model size | `runs/<size>` |
| `SAE_STATS` | per-domain statistics, `<width>/<size>/L<layer>.npz` | `stats/` |
| `SAE_WORK` | intermediate results shared between scripts | `work/` |
| `SAE_FIGS` | figures | `figures/` |
| `SAE_PROMPTS` | prompt CSV(s), comma-separated | `thesis/data/prompts.csv` |

Two modules are shared by many scripts. `agg_src.py` loads the statistics of one size and layer and keeps the 28
thesis domains. `shards.py` reads a capture per token, in the per-domain, per-prompt layout the per-token scripts use.

Script names keep the names they had during the thesis work, so that version suffixes (`v2`, `v3`, `_v6`) match
the file names they read and write. The order below is the order to run them in.

Memory: the per-token scripts that read the whole 27B capture of one layer need about 60 GB of RAM; nothing in
this folder needs a GPU except the local judge.

## 0. Prerequisites

```
python thesis/data/build_dataset.py --user-agent "your-project/1.0 (https://example.org/contact)"   # thesis/data/prompts.csv
python scripts/capture.py   --model 27b --prompts thesis/data/prompts.csv --out runs/27b        # and 270m, 1b, 4b, 12b
python scripts/aggregate.py --capture runs/27b --prompts thesis/data/prompts.csv --out stats/262k/27b
python scripts/capture.py   --model 27b --prompts thesis/data/prompts.csv --width 16k --out runs/27b_16k   # second dictionary control
python scripts/aggregate.py --capture runs/27b_16k --prompts thesis/data/prompts.csv --out stats/16k/27b
python thesis/data/make_lenmatched.py ...   # length-matched control; capture and aggregate it into stats/262k/<size>_lenm
python thesis/data/dataset_stats.py     # the dataset table
```

## 1. Always-on features (`analysis/always_on/`)

| script | output |
|---|---|
| `rate_mass_table.py` | mass by corpus firing-rate band per layer of 27B (`work/rate_mass_27b.json`) |
| `supp_machinery.py` | always-on features and their mass share, all sizes (`work/supp_machinery_v2.json`) |
| `fig_alwayson.py`, `v4_supp.py` | the always-on figures (need `feat_meta.py` from step 2) |

## 2. Feature classes (`analysis/classes/`)

Every fired feature of the 27B model gets one of five classes, position, token, word, concept or context, from a
classifier that reads the statistics of the token strings the feature fires on.

**With the shipped classifier** (`analysis/classes/classifier/feature_classifier.joblib`, the model of the thesis):

```
python thesis/analysis/classes/tf_feats.py 0 62                  # token statistics of every feature (split the range over jobs: 0 8, 8 16, ...)
python thesis/analysis/classes/feature_classifier.py apply 0 62  # work/tf_class_v6/L<L>.npz: the class of every fired feature
python thesis/analysis/classes/bridge_v4.py 0 62                 # per domain pair: shared features by class (work/bridge_v4_v6_v2/)
python thesis/analysis/classes/dom_profile.py; python thesis/analysis/classes/cummass.py; python thesis/analysis/classes/feat_meta.py
```

Always-on features (corpus rate at least .6) and rare features (fewer than 30 firings) are set aside as bands of their
own before the classes are counted. Figures: `fig_coverage.py`, `fig_h1_mass.py`, `v5_hinge.py`.

**How the classifier was made.** Two LLM judges labelled feature cards; the classifier was trained on those labels.
The scripts are included so the procedure can be repeated with new labels:

1. Cards from the parallel sets: `tokdump.py` and `heur_shared.py` (shared features of the parallel sets), then
   `v2_ex_sel.py` → `pilot_ex_dump.py` → `pilot_ex_cards.py` and `v3_ex_sel.py` → `pilot_ex_dump_v3.py` →
   `pilot_ex_cards_v3.py`. The reference judge (Claude Sonnet) read these cards with `judge/protocol_cards.txt`;
   `v2_labels_assemble.py` and `v3_labels_assemble.py` collect its answers.
2. Cards from every domain, judged by the local judge (Qwen3.5-122B-A10B through vLLM, `slurm/judge_lumi.sh`):
   ```
   LAYERS=0,2,4,6,7,12,15,18,21,24,27,30,32,38,43,47,52,58,61 OUT=work/dom_sel2.json python thesis/analysis/classes/dom_sel.py
   SEL=work/dom_sel2.json OUT=work/dom_dump2.json python thesis/analysis/classes/dom_dump.py
   TAG=2 python thesis/analysis/classes/dom_cards.py
   CARD_DIRS=work/dom_cards2 OUT=work/qwen_dom2_labels.jsonl PROTOCOL=thesis/analysis/classes/judge/protocol_qwen_v6.txt python thesis/analysis/classes/judge_vllm.py
   ```
3. Training table and model: `tf_extract.py`, `IDS=work/dom_cards2/index.json OUT=work/tf_dom2.json python
   thesis/analysis/classes/tf_extract.py`, `tf_train_export.py`, then `python thesis/analysis/classes/feature_classifier.py train work/tf_train.json`.
   The thesis model was trained on 5,401 judged features; its inner cross-validated accuracy against the judges is .76.
4. Checks: `judge_agree.py` (the two judges on the same cards) and `dom2_eval.py` (the classifier against the
   domain-wide labels).

The judge labels themselves are not distributed.

## 3. Feature sharing between domains (`analysis/sharing/`)

| script | measures |
|---|---|
| `ent_v4.py <size>` | Jaccard of the domain feature sets, every pair and layer, several set sizes |
| `ent_ctl2.py <size>`, `ent_ctl2b.py <size>` | split-half and Wikipedia baselines |
| `ent_mass.py <size> [16k]` | the mass the shared features carry |
| `ent_mass_cls.py 27b` | the same by class (needs step 2) |

Figures: `fig_52.py`, `fig_52_cells.py`, `fig_count_mass.py`, `fig_controls.py`, `a2_split.py`, `v4_story.py`.
Model sizes: run `ent_v4.py`, `ent_ctl2.py` and `ent_ctl2b.py` for every size, then `fig_size_sharing.py`,
`fig_size_kinds.py`, `fig_size_extra.py`, `fig_a3_ladder.py`.

## 4. Topic features and the gallery (`analysis/topics/`)

1. `hunt_stats_sizes.py <size>` for each of the five sizes — per-feature counts at eight layers per size.
2. `topic_super.py` — candidates under a looser rule, so passages exist for every feature the final rule keeps.
3. `topic_passages3.py <size>` — passages for the candidates.
4. `topic_sizes_eval.py` — the rule, the funnel per size and layer, agreement with the classifier, the matrices.
5. `build_topic_dataset3.py` — the topic-feature dataset (jsonl, csv index, readme).
6. `top1_hunt.py` then `fig_ladder.py` — all fired 27B features at the eight layers in eight groups.
7. The gallery (`gallery/` at the top level, published with GitHub Pages): `hunt_spill.py <L>` for the eight 27B
   layers 31, 35, ..., 59, then `python gallery/build_gallery.py` (writes `gallery/site/`).

## Differences from the thesis runs

- The thesis captures of the per-token statistics (steps 2 and 5) were made with an earlier PyTorch capture loop
  that also held the four other Wikipedia language domains. These scripts read the `saepipe` captures of the 28
  domains instead. Statistics that pool over domains, such as a feature's corpus rate inside `tf_feats.py`, shift
  slightly as a result, and the shipped classifier was trained on statistics of that earlier capture.
- The activation value is `pre * 1[pre > theta]` throughout, as in Gemma Scope.
- The Swedish diverse-language domain is now built like the other four (`disjoint_lang_swedish`, see
  `thesis/data/README.md`); the thesis used Swedish Wikipedia paragraphs (`div_lang_swedish`). Set
  `SAE_SWEDISH_DIVERSE=div_lang_swedish` to run the scripts on data with the thesis name.

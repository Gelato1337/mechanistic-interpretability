"""paths.py — every location the analysis scripts read or write. Set the environment variables to move them.

    SAE_RUNS     capture directories, one per model size: $SAE_RUNS/<size>      (scripts/capture.py --out)
                 or SAE_CAPTURE_<SIZE> for one size, e.g. SAE_CAPTURE_27B=/data/runs/27b_all (comma-separated if in parts)
    SAE_STATS    per-domain statistics: $SAE_STATS/<width>/<size>/L<layer>.npz (scripts/aggregate.py --out)
    SAE_WORK     intermediate results shared between the analysis scripts (json, npz, csv)
    SAE_FIGS     figures
    SAE_PROMPTS  comma-separated prompt CSV(s); default thesis/data/prompts.csv
    SAE_SWEDISH_DIVERSE  name of the Swedish diverse-language domain (default disjoint_lang_swedish)
Defaults are folders inside the repository (all git-ignored)."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]   # thesis/analysis/lib/paths.py -> repository root
RUNS = os.environ.get("SAE_RUNS", str(REPO / "runs"))
STATS = os.environ.get("SAE_STATS", str(REPO / "stats"))
WORK = os.environ.get("SAE_WORK", str(REPO / "work"))
FIGS = os.environ.get("SAE_FIGS", str(REPO / "figures"))
DATA = str(REPO / "thesis" / "data")
PROMPTS = os.environ.get("SAE_PROMPTS", f"{DATA}/prompts.csv").split(",")
SIZES = ["270m", "1b", "4b", "12b", "27b"]
# The diverse-language tier: disjoint_lang_<language> for all five languages (thesis/data/build_dataset.py). The thesis
# capture used a Swedish set of plain Wikipedia paragraphs instead, div_lang_swedish; set SAE_SWEDISH_DIVERSE to that
# name to run the scripts on the thesis data.
SWEDISH_DIVERSE = os.environ.get("SAE_SWEDISH_DIVERSE", "disjoint_lang_swedish")
DIVERSE_LANG = [f"disjoint_lang_{l}" for l in ("english", "finnish", "french", "german")] + [SWEDISH_DIVERSE]
UNUSED_DIVERSE = ({f"div_lang_{l}" for l in ("english", "finnish", "french", "german", "swedish")} | {"disjoint_lang_swedish"}) - set(DIVERSE_LANG)
MODEL_REPOS = {"270m": "google/gemma-3-270m", "1b": "google/gemma-3-1b-pt", "4b": "google/gemma-3-4b-pt",
               "12b": "google/gemma-3-12b-pt", "27b": "google/gemma-3-27b-pt"}


def capture(size):
    """Capture directory of one model size."""
    return capture_dirs(size)[0]


def capture_dirs(size):
    """Capture directories of one model size (SAE_CAPTURE_<SIZE> may list several, comma-separated, when the prompt set
    was captured in parts; the scripts that read shards treat them as one prompt list)."""
    return os.environ.get(f"SAE_CAPTURE_{size.upper()}", f"{RUNS}/{size}").split(",")


def stats_file(size, L, width="262k", extra=None):
    """Statistics of one layer; `extra` names a separately captured prompt set (e.g. the length-matched control)."""
    return f"{STATS}/{width}/{size}{'_' + extra if extra else ''}/L{L}.npz"


for _d in [WORK, FIGS] + [f"{WORK}/{s}" for s in ("tf_feats", "tf_class_v6", "hunt3", "topic_passages3", "deep", "pilot_cards", "spill_pages")] + \
          [f"{FIGS}/{s}" for s in ("act0", "act1", "act2", "act3", "hinge", "supp", "spill")]:
    os.makedirs(_d, exist_ok=True)

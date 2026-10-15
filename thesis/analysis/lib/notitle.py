"""notitle.py — import before drawing: drops figure suptitles, long axes titles and long fig.text headers (> 32 characters) so that a
thesis figure carries no header; panel labels such as "L24" or "27B · 62 layers" stay. Set NOTITLE=0 to keep the headers."""
import os, matplotlib.figure, matplotlib.axes
if os.environ.get("NOTITLE", "1") == "1":
    LIM = 32
    matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None
    _st = matplotlib.axes.Axes.set_title
    matplotlib.axes.Axes.set_title = lambda self, label="", *a, **k: (_st(self, label, *a, **k) if len(str(label)) <= LIM else None)
    _ft = matplotlib.figure.Figure.text
    matplotlib.figure.Figure.text = lambda self, x, y, s, *a, **k: (_ft(self, x, y, s, *a, **k) if len(str(s)) <= LIM else None)

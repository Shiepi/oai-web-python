# test_figs.py  (put in the same folder that contains the "app" folder)
from utils.cluster_analysis import build_cluster_figures

print("🟡 building figures… (may take 10–30 s)")
figs = build_cluster_figures()           # crunches t-SNE, UMAP, etc.

print("✅ got keys:", figs.keys())        # should list elbow, tsne, umap, sentiment
figs["tsne"].show()                      # opens your default browser

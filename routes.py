from flask import Blueprint, render_template
from utils.cluster_analysis  import build_cluster_figures
from utils.video_analysis    import build_video_figures  

bp = Blueprint("main", __name__)

@bp.route("/results")          # existing image-ads page
def results():
    return _render_results(build_cluster_figures(), video_mode=False)

@bp.route("/results/video")    # NEW video-ads page
def results_video():
    return _render_results(build_video_figures(),  video_mode=True)

def _render_results(figs: dict, *, video_mode: bool):
    html = {k: f.to_html(full_html=False, include_plotlyjs=False)
            for k, f in figs.items()}
    html["video_mode"] = video_mode
    return render_template("results_video.html", **html)

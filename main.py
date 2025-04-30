from flask import Flask, render_template
from plotly.utils import PlotlyJSONEncoder 
import plotly.express as px
import json
from utils.cluster_analysis import build_cluster_figures 
from utils.video_analysis   import build_video_figures

app = Flask(__name__, template_folder="templates", static_folder="static")

# Home page --------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", title="Home")

# About page -------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html", title="About")

# Analysis page (creates a Plotly chart) ---------------------
#@app.route("/analysis")
#def analysis():
    df  = px.data.iris()
 #   fig = px.scatter(
 #       df, x="sepal_width", y="sepal_length", color="species",
 #       title="Iris Sepal Dimensions", template="plotly_dark"  # <= dark template
 #  )
 #   graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
 #   return render_template("analysis.html", title="Analysis", graphJSON=graph_json)

@app.route("/analysis")
def analysis():
    return render_template("analysis.html", title="Analysis") 

@app.route("/results")
def results():
    figs = build_cluster_figures()                         # elbow, tsne, umap, sentiment

    # Convert each Plotly Figure to a tiny HTML snippet
    fig_html = {name: fig.to_html(
                           full_html=False,
                           include_plotlyjs=False)        # we load JS once in template
                for name, fig in figs.items()}

    return render_template("results.html", title="Results (Image Ads)", **fig_html)


@app.route("/results/video")        # NEW video-ads page
def results_video():
    figs = build_video_figures()
    html = {k: f.to_html(full_html=False, include_plotlyjs=False)
            for k,f in figs.items()}
    html["video_mode"] = True       # lets template show a badge, etc.
    return render_template("results_video.html", title="Results (Video Ads)" **html)  # or results.html if you reuse


if __name__ == "__main__":
    app.run(debug=True)
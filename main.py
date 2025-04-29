from flask import Flask, render_template
from plotly.utils import PlotlyJSONEncoder 
import plotly.express as px
import json

app = Flask(__name__)

# Home page --------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html", title="Home")

# About page -------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html", title="About")

# Analysis page (creates a Plotly chart) ---------------------
@app.route("/analysis")
def analysis():
    df  = px.data.iris()
    fig = px.scatter(
        df, x="sepal_width", y="sepal_length", color="species",
        title="Iris Sepal Dimensions", template="plotly_dark"  # <= dark template
    )
    graph_json = json.dumps(fig, cls=PlotlyJSONEncoder)
    return render_template("analysis.html", title="Analysis", graphJSON=graph_json)

# Results page ----------------------------------------------
@app.route("/results")
def results():
    # Placeholder – swap in real results
    return render_template("results.html", title="Results")

# Local dev convenience -------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=8080)

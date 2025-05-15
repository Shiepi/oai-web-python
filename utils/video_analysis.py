# utils/video_analysis.py
from pathlib import Path
import gcsfs
import pandas as pd, plotly.express as px
from plotly.graph_objs import Figure


fs = gcsfs.GCSFileSystem(token='google_default', project='cs163-optadpct')


# RAW_DIR = Path(r"C:\")
RAW_DIR = "cs163-optadpct.appspot.com/annotations_videos/video/"
PROCESSED = Path("processed_video_temp.csv")                                   # optional cache

def build_video_figures(force: bool = False) -> dict[str, Figure]:
    """Build all Plotly figures for *video ads* and return them in a dict."""

    # ── cache: reuse csv if already built ─────────────────────────────
    if PROCESSED.exists() and not force:
        merged_df = pd.read_csv(PROCESSED)
    else:
        # 1. ------------- your original load & merge code -------------
        file_path = "cs163-optadpct.appspot.com/annotations_videos/video/"

        with fs.open(file_path + "final_video_id_list.csv") as f:
            video_id_list = pd.read_csv(f, header=None, names=["video_id"])

        with fs.open(file_path + "topics_list.csv") as f:
            video_topics_list = pd.read_csv(f)

        with fs.open(file_path + "sentiments_list.csv") as f:
            video_sentiments_list = pd.read_csv(f)

        cleaned_path = file_path + "cleaned_result/"

        with fs.open(cleaned_path + "video_Effective_clean.json") as f:
            video_effective = pd.read_json(f, typ="series")

        with fs.open(cleaned_path + "video_Exciting_clean.json") as f:
            video_exciting = pd.read_json(f, typ="series")

        with fs.open(cleaned_path + "video_Funny_clean.json") as f:
            video_funny = pd.read_json(f, typ="series")

        with fs.open(cleaned_path + "video_Language_clean.json") as f:
            video_language = pd.read_json(f, typ="series")

        with fs.open(cleaned_path + "video_Topics_clean.json") as f:
            video_topics = pd.read_json(f, typ="series")

        with fs.open(cleaned_path + "video_Sentiments_clean.json") as f:
            video_sentiments = pd.read_json(f, typ="series")

        # -- index fix / DataFrame reshaping identical to your snippet --
        video_topics_list.index     += 1
        video_sentiments_list.index += 1
        video_topics_list     = video_topics_list.reset_index().rename(columns={"index":"topics_list_index"})
        video_sentiments_list = video_sentiments_list.reset_index().rename(columns={"index":"sentiments_list_index"})

        def to_df(s, name): 
            df = s.reset_index(); df.columns = ["video_id", name]; return df

        merged_df = (
            to_df(video_effective, "effective_score")
            .merge(to_df(video_exciting, "exciting_score"),  on="video_id")
            .merge(to_df(video_funny,    "funny_score"),     on="video_id")
            .merge(to_df(video_language, "language_score"),  on="video_id")
            .merge(to_df(video_sentiments,"sentiments"),     on="video_id")
            .merge(to_df(video_topics,   "topics"),          on="video_id")
        )

        merged_df = (merged_df
                     .merge(video_topics_list,     left_on="topics",     right_on="topics_list_index")
                     .merge(video_sentiments_list, left_on="sentiments", right_on="sentiments_list_index"))

        # keep only topics that occur > 30 times
        common_topics = merged_df["Topic"].value_counts()
        common_topics = common_topics[common_topics > 30].index          # <- fix
        merged_df = merged_df[merged_df["Topic"].isin(common_topics)]

        # keep only sentiments that occur > 30 times
        common_sentiments = merged_df["Sentiment"].value_counts()
        common_sentiments = common_sentiments[common_sentiments > 30].index
        merged_df = merged_df[merged_df["Sentiment"].isin(common_sentiments)]

        merged_df.to_csv(PROCESSED, index=False)

    # 2. ------------- build Plotly figures ---------------------------
    figs: dict[str, Figure] = {}

    # topic vs scores
    figs["topic_eff"] = px.box(merged_df, x="effective_score", y="Topic",
                               title="Effective Score by Topic")
    figs["topic_exc"] = px.box(merged_df, x="exciting_score",  y="Topic",
                               title="Exciting Score by Topic")
    figs["topic_fun"] = px.box(merged_df, x="funny_score",     y="Topic",
                               title="Funny Score by Topic")

    # sentiment vs scores
    figs["sent_eff"]  = px.box(merged_df, x="effective_score", y="Sentiment",
                               title="Effective Score by Sentiment")
    figs["sent_exc"]  = px.box(merged_df, x="exciting_score",  y="Sentiment",
                               title="Exciting Score by Sentiment")
    figs["sent_fun"]  = px.box(merged_df, x="funny_score",     y="Sentiment",
                               title="Funny Score by Sentiment")

    # language bars
    lang_by_topic = (merged_df.groupby(["Topic","language_score"])
                                .size().reset_index(name="Count"))
    figs["topic_lang"] = px.bar(lang_by_topic, x="Topic", y="Count",
                                color="language_score",
                                title="Language Score Distribution by Topic")

    lang_by_sent  = (merged_df.groupby(["Sentiment","language_score"])
                                .size().reset_index(name="Count"))
    figs["sent_lang"]  = px.bar(lang_by_sent, x="Sentiment", y="Count",
                                color="language_score",
                                title="Language Score Distribution by Sentiment")

    # stacked sentiment × topic
    st_counts = (merged_df.groupby(["Topic","Sentiment"])
                            .size().reset_index(name="count")
                            .sort_values("count", ascending=False))
    figs["topic_sent"] = px.bar(st_counts, x="Topic", y="count",
                                color="Sentiment", barmode="stack",
                                title="Sentiment Distribution Across Topics")

    return figs

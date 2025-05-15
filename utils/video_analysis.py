# utils/video_analysis.py
from pathlib import Path
import pandas as pd, plotly.express as px
from plotly.graph_objs import Figure

#RAW_DIR = Path(r"C:\")
RAW_DIR = Path("gs://cs163-optadpct.appspot.com/annotations_videos/video/")  
PROCESSED = RAW_DIR / "processed_video_temp.csv"                                   # optional cache

def build_video_figures(force: bool = False) -> dict[str, Figure]:
    """Build all Plotly figures for *video ads* and return them in a dict."""

    # ── cache: reuse csv if already built ─────────────────────────────
    if PROCESSED.exists() and not force:
        merged_df = pd.read_csv(PROCESSED)
    else:
        # 1. ------------- your original load & merge code -------------
        file_path = RAW_DIR.as_posix() + "/"

        video_id_list         = pd.read_csv(file_path + "final_video_id_list.csv",
                                            header=None, names=["video_id"])
        video_topics_list     = pd.read_csv(file_path + "topics_list.csv")
        video_sentiments_list = pd.read_csv(file_path + "sentiments_list.csv")

        video_effective   = pd.read_json(file_path + "cleaned_result/video_Effective_clean.json",  typ="series")
        video_exciting    = pd.read_json(file_path + "cleaned_result/video_Exciting_clean.json",   typ="series")
        video_funny       = pd.read_json(file_path + "cleaned_result/video_Funny_clean.json",      typ="series")
        video_language    = pd.read_json(file_path + "cleaned_result/video_Language_clean.json",   typ="series")
        video_topics      = pd.read_json(file_path + "cleaned_result/video_Topics_clean.json",     typ="series")
        video_sentiments  = pd.read_json(file_path + "cleaned_result/video_Sentiments_clean.json", typ="series")

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

        positive_sentiments = ['Inspired', 'Active', 'Amused', 'Amazed', 'Proud', 'Hopeful', 'Empowered', 'Excited', 'Cheerful', 'Youthful', 'Educated', 'Eager', 'Confident', 'Creative', 'Persuaded', 'Fashionable']
        negative_sentiments = ['Angry', 'Annoyed', 'Disgusted', 'Sad', 'Frustrated', 'Horrified', 'Alarmed']
        neutral_sentiments = ['Indifferent', 'Neutral', 'Calm', 'Relaxed', 'Alert', 'Conscious']

        # Function to classify sentiment
        def classify_sentiment(sentiment):
            if sentiment in positive_sentiments:
                return 'positive'
            elif sentiment in negative_sentiments:
                return 'negative'
            elif sentiment in neutral_sentiments:
                return 'neutral'
            else:
                return 'unknown'

        merged_df['sentiment_type'] = merged_df['Sentiment'].apply(classify_sentiment)

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

    figs["sent_type_effective"] = px.box(
        merged_df,
        x="sentiment_type",
        y="effective_score",
        color="sentiment_type",
        title="Distribution of Effective Scores by Sentiment Type",
        labels={"sentiment_type": "Sentiment Type", "effective_score": "Effective Score"}
    )

    return figs

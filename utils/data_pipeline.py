from pathlib import Path
import pandas as pd

# ───────────────────────────────────────────────────────────────
# CSV / JSON files
file_root    = Path("gs://cs163-optadpct.appspot.com/annotations_images/image")  
# file_root = Path(r"C:\")

# temp2.csv
# processed_path = Path(r"C:\")
processed_path = Path("gs://cs163-optadpct.appspot.com/processed/temp2.csv")
# ─────────────────────────────────────────────────────────────────────


import pandas as pd
import plotly.express as px

# file_path = Path(r"C:\")
file_path = "gs://cs163-optadpct.appspot.com/annotations_images/image"

image_topics_list = pd.read_csv(file_path + "topics_list.csv")
image_sentiments_list = pd.read_csv(file_path + "sentiments_list.csv")
image_strategies_list = pd.read_csv(file_path + "strategies_list.csv")

image_topics = pd.read_json(file_path + "Topics.json", typ="series")
image_sentiments = pd.read_json(file_path + "Sentiments.json", typ="series")
image_strategies = pd.read_json(file_path + "Strategies.json", typ="series")
image_symbols = pd.read_json(file_path + "Symbols.json", typ="series")

image_topics_list.index = image_topics_list.index + 1
image_sentiments_list.index = image_sentiments_list.index + 1
image_strategies_list.index = image_strategies_list.index + 1
image_topics_list = image_topics_list.reset_index()
image_sentiments_list = image_sentiments_list.reset_index()
image_strategies_list = image_strategies_list.reset_index()
image_topics_list = image_topics_list.rename(columns={"index": "topics_list_index"})
image_sentiments_list = image_sentiments_list.rename(columns={"index": "sentiments_list_index"})
image_strategies_list = image_strategies_list.rename(columns={"index": "strategies_list_index"})

image_topics_df = image_topics.reset_index()
image_topics_df.columns = ["image_id", "topics"]

image_sentiments_df = image_sentiments.reset_index()
image_sentiments_df.columns = ["image_id", "sentiments"]

image_strategies_df = image_strategies.reset_index()
image_strategies_df.columns = ["image_id", "strategies"]

image_symbols_df = image_symbols.reset_index()
image_symbols_df.columns = ["image_id", "symbols"]

merged_df = image_topics_df.merge(image_sentiments_df, on="image_id").merge(image_strategies_df, on="image_id").merge(image_symbols_df, on="image_id")

def extract_labels(symbols):
    return [s[-1] for s in symbols if isinstance(s, list)]
merged_df['symbol_labels'] = merged_df['symbols'].apply(extract_labels)

merged_df["sentiments_flat"] = merged_df["sentiments"].apply(lambda x: [item for sublist in x for item in sublist])
sentiments_flat_df = merged_df.explode("sentiments_flat")

merged_df["strategies_flat"] = merged_df["strategies"].apply(lambda x: [item for sublist in x for item in sublist])
sentiments_flat_df = merged_df.explode("strategies_flat")

merged_df["topics"] = merged_df["topics"].apply(lambda x: list(set(x)))
merged_df["sentiments_flat"] = merged_df["sentiments_flat"].apply(lambda x: list(set(x)))
merged_df["strategies_flat"] = merged_df["strategies_flat"].apply(lambda x: list(set(x)))
merged_df["symbol_labels"] = merged_df["symbol_labels"].apply(lambda x: list(set(x)))

clean_merged_df = merged_df[["image_id", "topics", "sentiments_flat", "strategies_flat", "symbol_labels"]].copy()

topic_map = image_topics_list.set_index('topics_list_index')['Category'].to_dict()
sentiment_map = image_sentiments_list.set_index('sentiments_list_index')['Sentiment'].to_dict()
strategy_map = image_strategies_list.set_index('strategies_list_index')['Strategy'].to_dict()

def map_topics(topic_list):
    return [topic_map.get(int(t), f"Unknown_{t}") if str(t).isdigit() else t for t in topic_list]

def map_sentiments(sentiment_list):
    return [sentiment_map.get(int(t), f"Unknown_{t}") if str(t).isdigit() else t for t in sentiment_list]

def map_strategies(strategy_list):
    return [strategy_map.get(int(t), f"Unknown_{t}") if str(t).isdigit() else t for t in strategy_list]

clean_merged_df['mapped_topics'] = clean_merged_df['topics'].apply(map_topics)
clean_merged_df['mapped_sentiments'] = clean_merged_df['sentiments_flat'].apply(map_sentiments)
clean_merged_df['mapped_strategies'] = clean_merged_df['strategies_flat'].apply(map_strategies)

clean_merged_df.to_csv("temp2.csv", index=False)

# print(image_topics_df, image_sentiments_df, image_strategies_df, image_symbols_df)
# print(image_topics_list)
# print(merged_df)

print(clean_merged_df)

# print(merged_df.dtypes)

# app/utils/cluster_analysis.py
import ast
import pandas as pd
import plotly.express as px
from plotly.graph_objs import Figure        # correct import for typing
from kmodes.kmodes import KModes
from sklearn.manifold import TSNE
import umap

# ──────────────────────────────────────────────────────────────
# ONE place to decide where temp2.csv lives:
processed_path = "gs://cs163-optadpct.appspot.com/processed/temp2.csv"
# processed_path = r"C:\"
# ──────────────────────────────────────────────────────────────


# utils/cluster_analysis.py
import ast, pandas as pd, plotly.express as px
from plotly.graph_objs import Figure
from kmodes.kmodes import KModes
from sklearn.manifold import TSNE
import umap

processed_path = "gs://cs163-optadpct.appspot.com/processed/temp2.csv"
#processed_path = r"C:\Users\Alice\Documents\SJSU\CS-163\processed\temp2.csv"  

def build_cluster_figures(k: int = 4) -> dict[str, Figure]:
    """
    Build and return all cluster-analysis plots.
    Keys: elbow, tsne, umap, sentiment,
          tsne_pos/neu/neg, umap_pos/neu/neg
    """
    # ───────────────── 1.  LOAD & PREP  ──────────────────────────
    full  = pd.read_csv(processed_path)
    clean = full.dropna().copy()
    for col in ['mapped_strategies', 'symbol_labels', 'mapped_sentiments']:
        clean[col] = clean[col].apply(ast.literal_eval)

    def binarize_top(df, col, top_n):
        exploded = df[[col]].explode(col)
        top = exploded[col].value_counts().head(top_n).index
        ohe = pd.get_dummies(exploded.loc[exploded[col].isin(top), col])
        ohe.index = exploded.loc[exploded[col].isin(top)].index
        return ohe.groupby(level=0).sum()

    strategy_ohe = binarize_top(clean, 'mapped_strategies', 10)
    symbol_ohe   = binarize_top(clean, 'symbol_labels',   20)
    features     = pd.concat([strategy_ohe, symbol_ohe], axis=1).fillna(0)

    # ───────────────── 2.  KModes elbow  ─────────────────────────
    Ks, cost = [], []
    for kk in range(1, 7):
        km = KModes(n_clusters=kk, init='random', n_init=5, verbose=0)
        km.fit_predict(features)
        Ks.append(kk); cost.append(km.cost_)

    fig_elbow = px.line(x=Ks, y=cost, markers=True,
                        labels=dict(x="k (clusters)", y="Cost"),
                        title="KModes Elbow Curve")

    # ───────────────── 3.  Final clustering  ─────────────────────
    kmodes   = KModes(n_clusters=k, init='random', n_init=5, verbose=0)
    clusters = kmodes.fit_predict(features)
    clean['Cluster'] = clusters

    # dominant sentiment helper
    pos = {'inspired','active','amused','amazed','proud','hopeful','empowered',
           'excited','cheerful','youthful','educated','eager','confident','creative',
           'persuaded','fashionable','empathetic','feminine','grateful','loving',
           'manly','thrifty'}
    neg = {'angry','annoyed','disgusted','sad','frustrated','horrified','alarmed',
           'afraid','disturbed','jealous','pessimistic'}
    neu = {'indifferent','neutral','calm','relaxed','alert','conscious','emotional'}

    def classify(s):
        s = s.lower().strip()
        if s in pos: return 'positive'
        if s in neg: return 'negative'
        if s in neu: return 'neutral'
        return 'unknown'

    exploded = clean.explode('mapped_sentiments').copy()
    exploded['sentiment_type'] = exploded['mapped_sentiments'].apply(classify)
    exploded['Cluster'] = exploded.index.map(clean['Cluster'])

    dominant_sent = (
        exploded.groupby(level=0)['sentiment_type']
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown')
    )

    # ───────────────── 4.  t-SNE & UMAP (3→2-D) ──────────────────
    tsne_df = pd.DataFrame(
        TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
            .fit_transform(features),
        columns=['TSNE-1','TSNE-2'])
    tsne_df['Cluster']   = clusters
    tsne_df['Index']     = tsne_df.index
    tsne_df['Sentiment'] = tsne_df.index.map(dominant_sent)
    tsne_df['mapped_topics'] =     clean.loc[tsne_df.index, 'mapped_topics']
    tsne_df['mapped_strategies'] = clean.loc[tsne_df.index, 'mapped_strategies']
    tsne_df['symbol_labels'] =     clean.loc[tsne_df.index, 'symbol_labels']

    umap_df = pd.DataFrame(
        umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1,
                  metric='hamming').fit_transform(features),
        columns=['UMAP-1','UMAP-2'])
    umap_df['Cluster']   = clusters
    umap_df['Index']     = umap_df.index
    umap_df['Sentiment'] = tsne_df['Sentiment']
    umap_df['mapped_topics'] =     clean.loc[umap_df.index, 'mapped_topics']
    umap_df['mapped_strategies'] = clean.loc[umap_df.index, 'mapped_strategies']
    umap_df['symbol_labels'] =     clean.loc[umap_df.index, 'symbol_labels']

    dominant_sentiment = exploded.groupby(exploded.index)['sentiment_type'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
    )

    def make_scatter(df, x, y, title):
        fig = px.scatter(df, x=x, y=y,
                         color='Cluster', symbol='Sentiment',
                         hover_data=['Cluster','Index','Sentiment', 'mapped_topics', 'Sentiment', 'mapped_strategies', 'symbol_labels'],
                         title=title)
        fig.update_traces(marker=dict(size=6, opacity=0.7))
        return fig

    fig_tsne = make_scatter(tsne_df, 'TSNE-1','TSNE-2', 't-SNE of KModes Clusters')
    fig_umap = make_scatter(umap_df, 'UMAP-1','UMAP-2', 'UMAP of KModes Clusters')

    # sentiment-filtered sub-plots
    fig_tsne_pos = make_scatter(tsne_df[tsne_df['Sentiment']=='positive'],
                                'TSNE-1','TSNE-2','t-SNE • Positive Sentiment')
    fig_tsne_neu = make_scatter(tsne_df[tsne_df['Sentiment']=='neutral'],
                                'TSNE-1','TSNE-2','t-SNE • Neutral Sentiment')
    fig_tsne_neg = make_scatter(tsne_df[tsne_df['Sentiment']=='negative'],
                                'TSNE-1','TSNE-2','t-SNE • Negative Sentiment')

    fig_umap_pos = make_scatter(umap_df[umap_df['Sentiment']=='positive'],
                                'UMAP-1','UMAP-2','UMAP • Positive Sentiment')
    fig_umap_neu = make_scatter(umap_df[umap_df['Sentiment']=='neutral'],
                                'UMAP-1','UMAP-2','UMAP • Neutral Sentiment')
    fig_umap_neg = make_scatter(umap_df[umap_df['Sentiment']=='negative'],
                                'UMAP-1','UMAP-2','UMAP • Negative Sentiment')

    # ───────────────── 5.  Sentiment-type bar  ────────────────────
    counts = (exploded.groupby(['Cluster','sentiment_type']).size()
                        .reset_index(name='count'))
    fig_bar = px.bar(counts, x='sentiment_type', y='count',
                     color='Cluster', barmode='group',
                     title='Cluster × Sentiment Type',
                     labels={'sentiment_type':'Sentiment','count':'Count'})
    fig_bar.update_layout(xaxis_tickangle=-45)

    # ───────────────── 6.  RETURN dict  ───────────────────────────
    return {
        "elbow": fig_elbow, "tsne": fig_tsne, "umap": fig_umap,
        "sentiment": fig_bar,
        "tsne_pos": fig_tsne_pos, "tsne_neu": fig_tsne_neu, "tsne_neg": fig_tsne_neg,
        "umap_pos": fig_umap_pos, "umap_neu": fig_umap_neu, "umap_neg": fig_umap_neg,
    }

# Optimizing Advertisement Impact (OAI)

**Authors: Alex Giang (alex.giang@sjsu.edu), Jeffrey Chan (jeffrey.n.chan@sjsu.edu)**

# Overview
A website that contains the analysis of the image & video ads dataset called, "Automatic Understanding of Image and Video Advertisements" by the University of Pittsburgh.
This project relies on the Image & Video Advertisements dataset introduced by Hussain et al. in their CVPR 2017 paper “Automatic Understanding of Image and Video Advertisements” (64832 images, 3477 videos, rich topic/sentiment/symbolism annotations). 


Please visit the [official dataset page](https://people.cs.pitt.edu/~kovashka/ads/) for license & citation details.

*A one-stop Flask/Plotly site for exploring how topic, sentiment, strategy, and symbolism shape the effectiveness of more than **64k+ image ads** and **3k+ video ads***\*.  

\*Dataset: “Automatic Understanding of Image and Video Advertisements,” Univ. of Pittsburgh (CVPR 2017).

[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](#) 

---

## Live Demo

➡️ **https://cs163-optadpct.appspot.com/**  
(Deployed via Google App Engine Standard)

---

## Quick Setup

### Required Packages
```bash
dash
plotly
pandas
Flask
gunicorn
google-cloud-storage
numpy
kmodes
umap-learn
gcsfs
scikit-learn
```

### Clone & install

```bash
git clone https://github.com/Shiepi/oai-web-python.git
cd oai-web-python
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```


## Pipeline
```bash
┌────────────┐       ┌────────────────────┐      ┌────────────────┐
│ Raw ads    │ ───► │ data_pipeline.py    │ ───► │ temp2.csv      │
│ annotations│       │  (merge + cleaning)│      │ (processed)    │
└────────────┘       └────────────────────┘      └────────────────┘
        │                                          │
        │                                          ▼
        │                               ┌────────────────────┐
        │                               │ cluster_analysis.py│───┐
        │                               │ (K-Modes + t-SNE   │   │Plotly
        │                               │  /UMAP figures)    │   │figs
        ▼                               └────────────────────┘   │
┌────────────────────┐                                  ┌────────▼────────┐
│ video_analysis.py   │ (box/stacked plots for video ads)│  Flask routes   │
│ (score & sentiment) │─────────────────────────────────►│ (main.py + Jinja)│
└────────────────────┘                                  └────────┬────────┘
                                                                 ▼
                                                       Deployed website

```

**Data Ingestion & Cleaning** – utils/data_pipeline.py reads the Image & Video Ads annotations (CSV/JSON) from the configured GCS bucket, normalises label indices and writes temp2.csv. 
GitHub

**Image-Ads Clustering** – utils/cluster_analysis.py loads temp2.csv, performs K-Modes clustering, dimensionality-reduces to 2-D with t-SNE/UMAP and returns nine Plotly figures (elbow, scatter maps, sentiment bars, etc.). 
GitHub

**Video-Ads Analytics** – utils/video_analysis.py builds topic/sentiment vs. effectiveness, humour, excitement and language-score plots for videos, caching an auxiliary processed_video_temp.csv. 
GitHub

**Web Layer** – main.py (or routes.py) calls those helpers, converts each Plotly figure to a snipped-HTML div, and injects them into Jinja templates under /results and /results/video. 
GitHub

**Deployment** – Served by Gunicorn behind App Engine; static assets (static/) are mapped via app.yaml. 


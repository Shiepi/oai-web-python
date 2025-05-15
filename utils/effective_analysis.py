import pandas as pd
import joblib
import plotly.express as px
from plotly.graph_objs import Figure
import plotly.figure_factory as ff
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────
# ONE place to decide where the processed data and model live:
processed_data_path = "gs://cs163-optadpct.appspot.com/processed/temp.csv"
# processed_data_path = r"C:\Users\Alice\Documents\SJSU\CS-163\processed\temp.csv"

model_path = "gs://cs163-optadpct.appspot.com/models/rf_pipeline.joblib"
# model_path = r"C:\Users\Alice\Documents\SJSU\CS-163\models\rf_pipeline.joblib"
# ──────────────────────────────────────────────────────────────

# Define feature and target columns consistent with training
FEATURE_COLUMNS = [
    'exciting_score', 'funny_score', 'language_score',
    'Topic', 'Sentiment', 'Description_topic', 'Description_sentiment',
    'Abbreviation_topic', 'Abbreviation_sentiment',
    'topics_list_index', 'sentiments_list_index', 'sentiment_type'
]
TARGET_COLUMN = 'effective_score'


def build_feature_importance_figure(n_top: int = 20) -> Figure:
    pipeline = joblib.load(model_path)
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    selector = pipeline.named_steps['selector']
    mask = selector.get_support()
    selected_features = feature_names[mask]
    importances = pipeline.named_steps['classifier'].feature_importances_

    df_imp = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(n_top)

    fig = px.bar(
        df_imp,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'Top {n_top} Feature Importances',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def evaluate_model_results(test_size: float = 0.2, random_state: int = 42) -> dict:
    data = pd.read_csv(processed_data_path)
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = joblib.load(model_path)
    y_pred = pipeline.predict(X_test)

    # Metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cls_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix heatmap
    labels = list(map(str, pipeline.named_steps['classifier'].classes_)) if hasattr(pipeline.named_steps['classifier'], 'classes_') else []
    fig_cm = ff.create_annotated_heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale='Viridis',
        showscale=True,
        annotation_text=cm.astype(str)
    )
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted label',
        yaxis_title='True label'
    )

    return {
        'balanced_accuracy': bal_acc,
        'classification_report': cls_report,
        'confusion_matrix_fig': fig_cm
    }

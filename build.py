from utils.data_pipeline import build_temp2_csv
csv_path = build_temp2_csv()          # builds once, returns the path
print("Cluster step will read:", csv_path)

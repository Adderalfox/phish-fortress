import os
import pandas as pd


def build_processed_features(
    dom_features_csv="../../data/html_dom_features_cleaned.csv",
    out_path="../../data/processed_features.csv"
):
    if not os.path.exists(dom_features_csv):
        raise FileNotFoundError(dom_features_csv)

    df = pd.read_csv(dom_features_csv)

    drop_cols = []
    for col in ["url", "Dest_port_anomalous", "Inbound_links_count", "Message_social_posts", "Diff_message_repeated", "html_path", "html_filename", "created_date", "rec_id"]:
        if col in df.columns:
            drop_cols.append(col)

    df = df.drop(columns=drop_cols, errors="ignore")

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in dataset")

    df.to_csv(out_path, index=False)
    print(f"Wrote processed features to {out_path}")
    return out_path


if __name__ == "__main__":
    build_processed_features()

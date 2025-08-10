import re
import os
import csv
import argparse
from typing import Optional


def find_html_paths(dataset_root: str):
    mapping = {}
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if f.lower().endswith(".html"):
                mapping[f] = os.path.join(root, f)
    return mapping


def parse_index_sql(index_sql_path: str):
    tuples = []
    with open(index_sql_path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    values_groups = re.findall(r"VALUES\s*\((.*?)\)\s*;", content, flags=re.IGNORECASE | re.DOTALL)
    for group in values_groups:
        parts = []
        cur = ""
        in_str = False
        esc = False
        for ch in group:
            if ch == "'" and not esc:
                in_str = not in_str
                cur += ch
            elif ch == "\\" and in_str:
                esc = not esc
                cur += ch
            elif ch == "," and not in_str:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
                esc = False
        if cur:
            parts.append(cur.strip())

        norm = []
        for p in parts:
            p = p.strip()
            if p.startswith("'") and p.endswith("'"):
                p = p[1:-1].replace("''", "'")
            norm.append(p)
        tuples.append(norm)
    return tuples


def extract_relevant_fields(tuple_parts):
    id_ = None
    url = ""
    html_fname = ""
    label = ""

    if len(tuple_parts) >= 1:
        if re.fullmatch(r"\d+", tuple_parts[0]):
            id_ = tuple_parts[0]

    for p in tuple_parts:
        if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://")):
            url = p
            break

    for p in tuple_parts:
        if isinstance(p, str) and re.search(r"\.html?$", p, flags=re.IGNORECASE):
            html_fname = os.path.basename(p)
            break

    for p in reversed(tuple_parts[-5:]):
        if p in ("0", "1"):
            label = int(p)
            break

    if not html_fname and id_:
        html_fname = f"{id_}.html"

    return id_, url, html_fname, label


def main(index_sql_path: str = "../data/index.sql", dataset_root: str = "../data/dataset", out_csv: str = "../data/index_parsed.csv"):
    html_map = find_html_paths(dataset_root)
    print(f"Found {len(html_map)} HTML files under {dataset_root}")

    tuples = parse_index_sql(index_sql_path)
    print(f"Parsed {len(tuples)} tuples from {index_sql_path}")

    with open(out_csv, "w", newline="", encoding="utf-8") as outf:
        writer = csv.writer(outf)
        writer.writerow(["id", "url", "html_filename", "label", "html_path"])
        for t in tuples:
            id_, url, html_fname, label = extract_relevant_fields(t)
            if not id_ and not url:
                continue
            html_path = html_map.get(html_fname, "")
            writer.writerow([id_ or "", url or "", html_fname or "", label if label != "" else "", html_path or ""])

    print(f"Wrote parsed index to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-sql", default="../data/index.sql")
    parser.add_argument("--dataset-root", default="../data/dataset")
    parser.add_argument("--out-csv", default="../data/index_parsed.csv")
    args = parser.parse_args()
    main(index_sql_path=args.index_sql, dataset_root=args.dataset_root, out_csv=args.out_csv)

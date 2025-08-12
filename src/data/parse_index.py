#!/usr/bin/env python3
import os
import re
import csv
import argparse
import html
from tqdm import tqdm

def build_html_map(dataset_root):
    html_map = {}
    for root, dirs, files in os.walk(dataset_root):
        for f in files:
            if f.endswith(".html"):
                html_map[f] = os.path.join(root, f)
    return html_map

def extract_insert_blocks_sqlsafe(content):
    blocks = []
    pattern = re.compile(
        r"INSERT\s+INTO\s+`?index`?(?:\s*\([^)]*\))?\s+VALUES\s*",
        flags=re.IGNORECASE
    )
    for m in pattern.finditer(content):
        start = m.end()
        i = start
        in_string = False
        escape_next = False
        while i < len(content):
            ch = content[i]
            if in_string:
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == "'":
                    in_string = False
            else:
                if ch == "'":
                    in_string = True
                elif ch == ";":
                    blocks.append(content[start:i])
                    break
            i += 1
    return blocks

def extract_tuples_sqlsafe(values_block):
    tuples = []
    current = []
    depth = 0
    in_string = False
    escape_next = False

    for ch in values_block:
        if in_string:
            current.append(ch)
            if escape_next:
                escape_next = False
            elif ch == "\\":
                escape_next = True
            elif ch == "'":
                in_string = False
        else:
            if ch == "'":
                in_string = True
                current.append(ch)
            elif ch == "(":
                if depth == 0:
                    current = ["("]
                else:
                    current.append(ch)
                depth += 1
            elif ch == ")":
                depth -= 1
                current.append(ch)
                if depth == 0:
                    tuples.append("".join(current))
                    current = []
            else:
                if depth > 0:
                    current.append(ch)
    return tuples

def parse_tuple(tup):
    s = tup.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    fields = []
    cur = []
    in_string = False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_string:
            if ch == "\\" and i + 1 < len(s):
                cur.append(s[i+1])
                i += 2
                continue
            if ch == "'" :
                if i + 1 < len(s) and s[i+1] == "'":
                    cur.append("'")
                    i += 2
                    continue
                # closing quote
                in_string = False
                i += 1
                continue
            else:
                cur.append(ch)
                i += 1
        else:
            if ch == "'":
                in_string = True
                i += 1
                continue
            elif ch == ",":
                fields.append(''.join(cur).strip())
                cur = []
                i += 1
                continue
            else:
                cur.append(ch)
                i += 1

    if cur:
        fields.append(''.join(cur).strip())

    processed = []
    for f in fields:
        if f.upper() == 'NULL':
            processed.append(None)
        elif len(f) >= 2 and f[0] == "'" and f[-1] == "'" :
            inner = f[1:-1]
            inner = inner.replace("\\'", "'").replace("''", "'").replace('\\\\', '\\')
            inner = html.unescape(inner)
            processed.append(inner)
        elif f.startswith("'") and f.endswith("'") is False and "'" in f:
            inner = f.replace("\\'", "'").replace("''", "'").replace('\\\\', '\\')
            inner = html.unescape(inner)
            processed.append(inner)
        else:
            processed.append(f if f != '' else None)
    return processed

def parse_sql(sql_file, html_map, out_csv):
    """Parse index.sql into a CSV with HTML paths resolved."""
    with open(sql_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    insert_blocks = extract_insert_blocks_sqlsafe(content)
    print(f"Found {len(insert_blocks)} INSERT blocks.")

    total_rows = 0
    missing_html = 0
    total_tuples_found = 0
    skipped_tuples = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["rec_id", "url", "html_filename", "label", "created_date", "html_path"])

        for block in tqdm(insert_blocks, desc="Parsing INSERT blocks"):
            tuples = extract_tuples_sqlsafe(block)
            total_tuples_found += len(tuples)
            for tup in tuples:
                fields = parse_tuple(tup)

                if len(fields) < 5:
                    skipped_tuples += 1
                    continue

                rec_id = fields[0]
                url = fields[1]
                html_fname = fields[2]
                label = fields[3]
                created_date = fields[4]

                html_path = html_map.get(html_fname, "")
                if not html_path:
                    missing_html += 1

                writer.writerow([rec_id, url, html_fname, label, created_date, html_path])
                total_rows += 1

    print(f"Parsed rows: {total_rows}")
    print(f"Missing HTML files: {missing_html}")
    print(f"Total tuples found: {total_tuples_found}")
    print(f"Tuples skipped (too few fields): {skipped_tuples}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse index.sql into CSV with HTML paths")
    parser.add_argument("--sql", required=True, help="Path to index.sql")
    parser.add_argument("--dataset-root", required=True, help="Path to dataset root folder containing dataset-part-* folders")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    html_map = build_html_map(args.dataset_root)
    print(f"Found {len(html_map)} HTML files in dataset parts.")

    parse_sql(args.sql, html_map, args.out)

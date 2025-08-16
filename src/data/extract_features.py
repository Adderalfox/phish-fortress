import os
import re
import math
import argparse
from collections import Counter, defaultdict
from urllib.parse import urlparse, urljoin, parse_qs

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


TAG_INTERACTIVE = {"a", "button", "input", "select", "textarea", "label", "details", "summary"}
RESOURCE_TAG_ATTR = {
    "script": "src",
    "img": "src",
    "iframe": "src",
    "link": "href",
    "audio": "src",
    "video": "src",
    "source": "src",
    "embed": "src",
    "object": "data",
}
DOWNLOAD_EXT = {".zip", ".rar", ".7z", ".exe", ".msi", ".apk", ".dmg", ".pdf", ".doc", ".docx", ".xls", ".xlsx"}

def norm_domain(u: str) -> str:
    try:
        p = urlparse(u)
        return (p.netloc or "").lower()
    except Exception:
        return ""

def is_httpish(u: str) -> bool:
    return u.startswith("http://") or u.startswith("https://")

def is_absolute(u: str) -> bool:
    return bool(urlparse(u).netloc)

def count_occurrences(text: str, patterns):
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)

def text_len(soup: BeautifulSoup) -> int:
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return len(soup.get_text(separator=" ", strip=True))

def dom_depth(el) -> int:
    if not hasattr(el, "children"):
        return 1
    depths = [1]
    for c in el.children:
        try:
            if getattr(c, "name", None):
                depths.append(1 + dom_depth(c))
        except RecursionError:
            continue
    return max(depths) if depths else 1

def avg_children_per_node(soup: BeautifulSoup) -> float:
    total_nodes, total_children = 0, 0
    for tag in soup.find_all(True):
        total_nodes += 1
        total_children += sum(1 for c in tag.children if getattr(c, "name", None))
    return total_children / total_nodes if total_nodes else 0.0

def has_meta_refresh(soup: BeautifulSoup) -> int:
    for m in soup.find_all("meta"):
        if m.get("http-equiv", "").lower() == "refresh":
            return 1
    return 0

def inline_event_handlers_count(soup: BeautifulSoup) -> int:
    cnt = 0
    for tag in soup.find_all(True):
        for attr in list(tag.attrs.keys()):
            if isinstance(attr, str) and attr.lower().startswith("on"):
                cnt += 1
    return cnt

def hidden_elements_count(soup: BeautifulSoup) -> int:
    cnt = 0
    for tag in soup.find_all(True):
        style = tag.get("style", "")
        if isinstance(style, list):
            style = " ".join(style)
        style_low = style.lower()
        if "display:none" in style_low or "visibility:hidden" in style_low or "opacity:0" in style_low:
            cnt += 1
        if tag.has_attr("hidden"):
            cnt += 1
    return cnt

def file_ext(path: str) -> str:
    try:
        base = urlparse(path).path
        return os.path.splitext(base)[1].lower()
    except Exception:
        return ""

def count_download_links(anchors) -> int:
    c = 0
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href:
            continue
        ext = file_ext(href)
        if ext in DOWNLOAD_EXT or a.has_attr("download"):
            c += 1
    return c

def count_mismatch_text_vs_href(anchors) -> int:
    patt = re.compile(r"(https?://)?([a-z0-9.-]+\.[a-z]{2,})", re.I)
    c = 0
    for a in anchors:
        txt = (a.get_text(" ", strip=True) or "")[:200]
        href = (a.get("href") or "")
        if not href:
            continue
        m = patt.search(txt)
        if m:
            shown_dom = norm_domain(m.group(0))
            href_dom = norm_domain(href)
            if shown_dom and href_dom and shown_dom != href_dom:
                c += 1
    return c


def extract_features(html_path: str, page_url: str):
    feats = defaultdict(lambda: math.nan)

    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml") if BeautifulSoup else BeautifulSoup(html, "html.parser")

    page_domain = norm_domain(page_url) if page_url else ""

    anchors = soup.find_all("a")
    forms = soup.find_all("form")
    scripts = soup.find_all("script")
    iframes = soup.find_all("iframe")
    links_link = soup.find_all("link")
    images = soup.find_all("img")
    metas = soup.find_all("meta")

    all_elements = soup.find_all(True)
    all_text = str(soup)

    anchor_total = len([a for a in anchors if a.get("href")])
    anchor_match = 0
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href:
            continue
        if not is_httpish(href):
            anchor_match += 1
        else:
            if page_domain and norm_domain(href) == page_domain:
                anchor_match += 1
    feats["Anchor_ratio_same_domain"] = anchor_match / anchor_total if anchor_total else 1.0

    ext_domains = set()
    for tag, attr in RESOURCE_TAG_ATTR.items():
        for el in soup.find_all(tag):
            src = (el.get(attr) or "").strip()
            if src and is_httpish(src):
                d = norm_domain(src)
                if d and d != page_domain:
                    ext_domains.add(d)
    feats["U_request_unique_ext_domains"] = len(ext_domains)

    feats["Popup_indicators"] = int(
        any(p.search(all_text) for p in [re.compile(r"\bwindow\.open\s*\("), re.compile(r"\balert\s*\("), re.compile(r"\bprompt\s*\(")])
    )

    bl_like = 0
    for el in links_link + metas + scripts:
        target = el.get("href") or el.get("content") or el.get("src") or ""
        if is_httpish(target):
            if page_domain and norm_domain(target) != page_domain:
                bl_like += 1
    feats["Links_in_tags_external_count"] = bl_like

    abnormal = 0
    for tag in ["object", "embed", "iframe"]:
        for el in soup.find_all(tag):
            src = el.get("data") or el.get("src") or ""
            if is_httpish(src) and page_domain and norm_domain(src) != page_domain:
                abnormal += 1
    feats["Abn_req_URL_count"] = abnormal

    feats["Cookies_script_usage"] = int(bool(re.search(r"\bdocument\.cookie\b", all_text)))

    cross_if = 0
    for ifr in iframes:
        src = (ifr.get("src") or "").strip()
        if is_httpish(src) and page_domain and norm_domain(src) != page_domain:
            cross_if += 1
    feats["Iframe_cross_domain_count"] = cross_if

    submit_cnt = 0
    submit_cnt += sum(1 for b in soup.find_all("button") if b.get("type", "").lower() == "submit")
    submit_cnt += sum(1 for i in soup.find_all("input") if i.get("type", "").lower() == "submit")
    feats["Submit_button_count"] = submit_cnt

    feats["Form_count"] = len(forms)

    fav_ext = 0
    for l in links_link:
        rel = " ".join(l.get("rel", [])).lower() if l.has_attr("rel") else ""
        if "icon" in rel:
            href = (l.get("href") or "").strip()
            if is_httpish(href) and page_domain and norm_domain(href) != page_domain:
                fav_ext += 1
    feats["Favicon_external_count"] = fav_ext

    feats["Mailto_link_count"] = sum(1 for a in anchors if (a.get("href") or "").lower().startswith("mailto:"))

    a_img = [a for a in anchors if a.find("img")]
    match_ai = 0
    for a in a_img:
        href = (a.get("href") or "")
        if not href:
            continue
        if not is_httpish(href) or (page_domain and norm_domain(href) == page_domain):
            match_ai += 1
    feats["IMG_Hyperlink_ratio_same_domain"] = (match_ai / len(a_img)) if a_img else 1.0

    feats["Susp_links_mismatch_text_vs_href"] = count_mismatch_text_vs_href(anchors)

    right_block = 0
    if re.search(r"oncontextmenu\s*=", all_text, flags=re.I):
        right_block = 1
    if re.search(r"document\.addEventListener\s*\(\s*['\"]contextmenu['\"]\s*,", all_text, flags=re.I):
        right_block = 1
    feats["Right_click_blocked"] = right_block

    https_forms = 0
    abs_forms = 0
    rel_forms = 0
    for f in forms:
        action = (f.get("action") or "").strip()
        if not action:
            rel_forms += 1
            continue
        if is_httpish(action):
            abs_forms += 1
            if action.startswith("https://"):
                https_forms += 1
        else:
            rel_forms += 1
    feats["Security_https_forms_ratio"] = (https_forms / abs_forms) if abs_forms else 1.0

    total_forms = abs_forms + rel_forms
    feats["Action_relative_ratio"] = (rel_forms / total_forms) if total_forms else 0.0

    feats["Dest_port_anomalous"] = math.nan
    feats["Inbound_links_count"] = math.nan
    feats["Message_social_posts"] = math.nan
    feats["Diff_message_repeated"] = math.nan

    feats["DOM_nodes_count"] = len(all_elements)
    feats["DOM_depth_max"] = dom_depth(soup.html or soup)
    tag_counts = Counter([t.name for t in all_elements if t.name])
    total_tags = sum(tag_counts.values()) or 1
    probs = [c / total_tags for c in tag_counts.values()]
    feats["Tag_entropy"] = -sum(p * math.log(p + 1e-12, 2) for p in probs)
    feats["Top_tag_ratio"] = (max(tag_counts.values()) / total_tags) if tag_counts else 0.0

    class_names = []
    for t in all_elements:
        cls = t.get("class")
        if isinstance(cls, list):
            class_names.extend([c for c in cls if isinstance(c, str)])
        elif isinstance(cls, str):
            class_names.append(cls)
    feats["CSS_class_unique_count"] = len(set(class_names))

    feats["DOM_element_types_count"] = len(set(tag_counts.keys()))

    feats["Text_total_length"] = text_len(soup)

    feats["Avg_children_per_node"] = avg_children_per_node(soup)

    feats["Hidden_form_inputs_count"] = sum(1 for i in soup.find_all("input") if i.get("type", "").lower() == "hidden")

    feats["Popup_windows_indicator"] = int(bool(re.search(r"\bwindow\.open\s*\(", all_text)))

    feats["Iframe_elements_count"] = len(iframes)

    ext_res = 0
    for tag, attr in RESOURCE_TAG_ATTR.items():
        for el in soup.find_all(tag):
            src = (el.get(attr) or "").strip()
            if is_httpish(src) and page_domain and norm_domain(src) != page_domain:
                ext_res += 1
    feats["External_resources_count"] = ext_res

    feats["Script_tags_count"] = len(scripts)

    feats["Meta_refresh_present"] = has_meta_refresh(soup)

    feats["Interactive_elements_count"] = sum(1 for t in all_elements if t.name in TAG_INTERACTIVE)

    mismatch = 0
    for a in anchors:
        href = (a.get("href") or "").strip()
        if is_httpish(href) and page_domain and norm_domain(href) != page_domain:
            mismatch += 1
    for f in forms:
        action = (f.get("action") or "").strip()
        if is_httpish(action) and page_domain and norm_domain(action) != page_domain:
            mismatch += 1
    feats["Mismatched_domains_count"] = mismatch

    feats["Hover_URLs_mismatch"] = feats["Susp_links_mismatch_text_vs_href"]

    feats["Download_links_count"] = count_download_links(anchors)

    mixed = 0
    page_https = (page_url or "").startswith("https://")
    if page_https:
        for tag, attr in RESOURCE_TAG_ATTR.items():
            for el in soup.find_all(tag):
                src = (el.get(attr) or "").strip().lower()
                if src.startswith("http://"):
                    mixed += 1
    feats["Mixed_content_http_on_https"] = mixed

    try:
        q = parse_qs(urlparse(page_url).query)
        feats["URL_params_count"] = sum(len(v) for v in q.values())
    except Exception:
        feats["URL_params_count"] = 0

    return dict(feats)


def process(index_csv: str, out_csv: str, start: int = 0, end: int | None = None):
    df = pd.read_csv(index_csv)
    if end is not None:
        df = df.iloc[start:end]
    else:
        df = df.iloc[start:]

    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        html_path = row.get("html_path", "")
        page_url = row.get("url", "")
        if not isinstance(html_path, str) or not html_path or not os.path.exists(html_path):
            continue
        try:
            feats = extract_features(html_path, page_url)
            row_dict = row.to_dict()
            row_dict.update(feats)
            results.append(row_dict)
        except Exception as e:
            results.append({**row.to_dict(), "feature_error": str(e)})

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)
    print(f"✅ Saved {len(out_df)} rows with features -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Table 2 & 3 HTML/DOM features from local HTML files")
    parser.add_argument("--index-csv", required=True, help="CSV with columns: url, html_path (plus label/result if available)")
    parser.add_argument("--out", default="html_dom_features.csv", help="Output CSV")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    process(args.index_csv, args.out, start=args.start, end=args.end)

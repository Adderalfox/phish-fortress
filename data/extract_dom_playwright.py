import pandas as pd
import argparse
import os
from playwright.sync_api import sync_playwright
from tqdm import tqdm

DEFAULT_OUT = "../data/dom_js_features.parquet"


def extract_features_from_page(page):
    script = """
    () => {
      const res = {};
      const doc = document;
      res['num_elements'] = doc.getElementsByTagName('*').length;
      res['num_iframes'] = doc.getElementsByTagName('iframe').length;
      res['num_forms'] = doc.getElementsByTagName('form').length;
      res['num_inputs'] = doc.querySelectorAll('input, textarea, select').length;
      res['num_scripts'] = doc.getElementsByTagName('script').length;
      // count external scripts (src not empty)
      const scripts = Array.from(doc.getElementsByTagName('script'));
      res['num_external_scripts'] = scripts.filter(s => s.src && s.src.length>0).length;
      // inline event handlers (onsubmit, onclick etc)
      const event_attrs = ['onclick','onmouseover','onerror','onsubmit','onload'];
      let inline_events = 0;
      Array.from(doc.getElementsByTagName('*')).forEach(el => {
        for (const a of event_attrs) {
          if (el.getAttribute && el.getAttribute(a)) inline_events++;
        }
      });
      res['inline_event_handlers'] = inline_events;
      // suspicious functions (eval, Function)
      const all_text = doc.documentElement.innerHTML;
      res['eval_count'] = (all_text.match(/\\beval\\s*\\(/g) || []).length;
      res['Function_usage'] = (all_text.match(/\\bnew\\s+Function\\s*\\(/g) || []).length + (all_text.match(/\\bFunction\\s*\\(/g) || []).length;
      // number of iframes with src pointing to other domains
      try {
        const iframes = Array.from(doc.getElementsByTagName('iframe'));
        let cross_iframes = 0;
        iframes.forEach(ifr => {
          try {
            const src = ifr.getAttribute('src') || '';
            if (src && !src.startsWith(location.origin) && src.includes('http')) cross_iframes++;
          } catch(e) {}
        });
        res['cross_domain_iframes'] = cross_iframes;
      } catch(e) {
        res['cross_domain_iframes'] = 0;
      }
      return res;
    }
    """
    return page.evaluate(script)


def process_rows(df, out_path, start_idx=0, end_idx=None, wait_until="load", timeout=30000):
    rows = df.iloc[start_idx:end_idx] if end_idx else df.iloc[start_idx:]
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(ignore_https_errors=True)
        page = context.new_page()
        for idx, row in tqdm(rows.iterrows(), total=len(rows)):
            html_path = row.get("html_path", "")
            row_id = row.get("id", "")
            if not html_path or not os.path.exists(html_path):
                results.append({**row.to_dict(), **{"error": "html_not_found"}})
                continue
            try:
                file_url = f"file://{os.path.abspath(html_path)}"
                page.goto(file_url, wait_until=wait_until, timeout=timeout)
                page.wait_for_load_state("networkidle", timeout=2000)
                metrics = extract_features_from_page(page)
                entry = {**row.to_dict(), **metrics}
                results.append(entry)
            except Exception as e:
                results.append({**row.to_dict(), **{"error": str(e)}})

        browser.close()

    out_df = pd.DataFrame(results)
    out_df.to_parquet(out_path, index=False)
    print(f"Written features to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", default="../data/index_parsed.csv")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    process_rows(df, args.out, start_idx=args.start, end_idx=args.end)

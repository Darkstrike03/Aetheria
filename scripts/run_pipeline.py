"""Run the data pipeline: scrape -> clean -> (optional) train prototype model.

Usage examples:
  # scrape using urls.txt and clean
  python scripts/run_pipeline.py

  # run scrape+clean and then train (calls prototype_model.main())
  python scripts/run_pipeline.py --train
"""

import argparse
from pathlib import Path
import importlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run prototype_model training after scraping+cleaning')
    parser.add_argument('--skip-scrape', action='store_true', help='Do not run scraper, only clean existing raw data')
    args = parser.parse_args()

    scripts_dir = Path(__file__).parents[0]
    # import modules by file path to avoid package/module name issues
    import importlib.util

    def load_module_from_path(name, path: Path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    scrape = load_module_from_path('scrape_data', scripts_dir / 'scrape_data.py')
    clean = load_module_from_path('clean_data', scripts_dir / 'clean_data.py')

    if not args.skip_scrape:
        urls_file = Path(__file__).parents[1] / 'urls.txt'
        urls = []
        if urls_file.exists():
            urls = [l.strip() for l in urls_file.read_text(encoding='utf-8').splitlines() if l.strip()]
        if urls:
            scrape.scrape_pages(urls)
        else:
            print('No urls.txt found or it is empty. Skipping scraping.')

    # clean raw text
    clean.main()

    if args.train:
        print('Starting prototype training (this may be slow).')
        proto = load_module_from_path('prototype_model', scripts_dir / 'prototype_model.py')
        # call main() from prototype_model which will train and save the model
        proto.main()


if __name__ == '__main__':
    main()

"""Builds the product-level external-data tables   features/external/{fk,fdc,off}.parquet   (one row per Instacart product; NaN = no match).
  cd src && python -m external.build [--only fk fdc off]
Needs dataset/external/{foodkeeper, fdc/branded, off}/  (see README).  Does NOT touch the 54 base features."""
import argparse
import importlib

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="+", default=["fk", "fdc", "off"], choices=["fk", "fdc", "off"])
    for name in ap.parse_args().only:
        importlib.import_module(f"external.{name}").build()

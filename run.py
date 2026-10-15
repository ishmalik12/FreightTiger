"""Usage: python run.py [--shipments data/shipment_records.csv] [--notes data/context_notes.csv] [--out outputs]"""
import argparse

from freightwatch.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shipments", default="data/shipment_records.csv")
    p.add_argument("--notes", default="data/context_notes.csv")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    out = run(a.shipments, a.notes, a.out)
    print(f"{len(out)} flagged-candidate route-weeks -> {a.out}/flagged_routes.csv")
    print(out.flagged.value_counts().to_string())

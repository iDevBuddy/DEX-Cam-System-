"""Rename a machine zone WITHOUT splitting its history.

Machine names live as plain strings in machine_states / machine_visits —
renaming a zone only in config.yaml would make reports show the old and
new name as two different machines. This tool renames the DB rows;
config.yaml ka naam alag se (hath se) badalna hai, phir server restart.

Usage (server ka chalna theek hai — WAL + 10s timeout, magar lakhon rows
nahi hain to turant ho jata hai):
    python tools/rename_machine.py factory-cam-4 top-machine lathe-big
    python tools/rename_machine.py factory-cam-4 top-machine lathe-big --dry-run
"""
import argparse
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("DEX_DB") or ROOT / "data" / "events.db")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("camera", help="camera name, e.g. factory-cam-4")
    ap.add_argument("old", help="current zone name in the DB")
    ap.add_argument("new", help="new machine name")
    ap.add_argument("--dry-run", action="store_true",
                    help="sirf ginti dikhao, kuch badlo mat")
    args = ap.parse_args()

    if not DB_PATH.exists():
        sys.exit(f"DB nahi mili: {DB_PATH}")

    con = sqlite3.connect(DB_PATH, timeout=10)
    try:
        counts = {}
        for table, col in (("machine_states", "machine"),
                           ("machine_visits", "machine"),
                           ("machine_visits", "switched_from")):
            (n,) = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE camera=? AND {col}=?",
                (args.camera, args.old)).fetchone()
            counts[f"{table}.{col}"] = n

        total = sum(counts.values())
        for k, v in counts.items():
            print(f"  {k:32s} {v} rows")
        if total == 0:
            print(f"'{args.old}' @ {args.camera} ki koi rows nahi — "
                  "naam check karo (ya pehli dafa hai, sirf config badlo).")
            return
        if args.dry_run:
            print(f"[dry-run] {total} rows '{args.old}' -> '{args.new}' hoti.")
            return

        with con:
            for table, col in (("machine_states", "machine"),
                               ("machine_visits", "machine"),
                               ("machine_visits", "switched_from")):
                con.execute(
                    f"UPDATE {table} SET {col}=? WHERE camera=? AND {col}=?",
                    (args.new, args.camera, args.old))
        print(f"DONE: {total} rows '{args.old}' -> '{args.new}' "
              f"({args.camera}).")
        print("Ab config.yaml mein zone ka naam badlo aur server restart karo.")
    finally:
        con.close()


if __name__ == "__main__":
    main()

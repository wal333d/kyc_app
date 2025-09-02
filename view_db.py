#!/usr/bin/env python3
import argparse, sqlite3, csv, sys

def main():
    p = argparse.ArgumentParser(description="Quick viewer/exporter for kyc_logs.db")
    p.add_argument("--db", default="kyc_logs.db")
    p.add_argument("--table", choices=["verified_people","logs"], default="verified_people")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--csv", help="Write CSV to this file instead of printing a table")
    args = p.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    rows = con.execute(f"SELECT * FROM {args.table} ORDER BY ts DESC LIMIT ?", (args.limit,)).fetchall()
    con.close()

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if rows:
                w.writerow(rows[0].keys())
                for r in rows:
                    w.writerow([r[k] for k in r.keys()])
        print(f"Wrote {len(rows)} rows to {args.csv}")
    else:
        if not rows:
            print("No rows.")
            return
        headers = list(rows[0].keys())
        widths = [max(len(h), *(len(str(r[h])) for r in rows)) for h in headers]
        fmt = " | ".join("{:"+str(w)+"}" for w in widths)
        print(fmt.format(*headers))
        print("-+-".join("-"*w for w in widths))
        for r in rows:
            print(fmt.format(*[str(r[h]) for h in headers]))

if __name__ == "__main__":
    main()

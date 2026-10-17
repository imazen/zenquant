#!/usr/bin/env python3
"""Paired log-Student intervals for the generated unused-report cost probe."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
from scipy.stats import t


def analyze(text):
    groups={}
    for kind,side,a,b in re.findall(r"^cost kind=(\d+) side=(\d+).*?candidate_ns=(\d+) before_ns=(\d+)",text,re.M):
        groups.setdefault((int(kind),int(side)),[]).append(math.log(int(a)/int(b)))
    if set(groups)!={(kind,side) for kind in (1,2) for side in (8,64,256,1024)}:
        raise ValueError("complete generated size/content grid required")
    out=[]
    for (kind,side),values in sorted(groups.items()):
        if len(values)!=20:raise ValueError("twenty paired rounds per cell required")
        mean=statistics.mean(values)
        half=t.ppf(.975,len(values)-1)*statistics.stdev(values)/math.sqrt(len(values))
        lo,hi=math.exp(mean-half),math.exp(mean+half)
        out.append(dict(kind=kind,side=side,ratio_geomean=math.exp(mean),paired95=[lo,hi],
                        measurable_increase=bool(lo>1)))
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",type=Path)
    args=parser.parse_args()
    rows=analyze(args.log.read_text())
    print(json.dumps(rows,indent=2))
    return int(any(r["measurable_increase"] for r in rows))


if __name__=="__main__":raise SystemExit(main())

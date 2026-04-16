# -*- coding: utf-8 -*-
"""
GSE175634 official-method benchmark entry.

Runs the official comparison pipeline for all settings or one selected setting:
- scLineagetracer
- CellRank (official)
- WOT (official)
- CoSpar (official)
"""

import argparse


def run_all(run_cospar: int):
    import compare_roc_official_alltime_remaining3 as alltime
    alltime.run_dataset("GSE175634", run_cospar=int(run_cospar))


def run_one(setting: str, run_cospar: int):
    import compare_roc_official_remaining3 as core
    import compare_roc_official_alltime_remaining3 as alltime

    out_root = "/Users/wanghongye/python/scLineagetracer/classification/GSE175634/GSE175634_CMvsCF/roc_click/benchmark_all_timepoints_official"
    metas = alltime.gen_meta_175634_all(out_root)
    meta_map = {m["setting"]: m for m in metas}
    if setting not in meta_map:
        raise ValueError(f"Unknown setting: {setting}. Available: {sorted(meta_map.keys())}")
    core.run_one(meta_map[setting], run_cospar=int(run_cospar))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "one"], default="all")
    parser.add_argument("--setting", choices=["Obs_Day1", "Obs_Day3", "Obs_Day5", "Obs_Day7", "Obs_Day11"], default="Obs_Day1")
    parser.add_argument("--run_cospar", type=int, default=1, help="1=run official CoSpar")
    args = parser.parse_args()

    if args.mode == "all":
        run_all(run_cospar=int(args.run_cospar))
    else:
        run_one(setting=str(args.setting), run_cospar=int(args.run_cospar))


if __name__ == "__main__":
    main()

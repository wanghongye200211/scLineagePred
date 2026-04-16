# -*- coding: utf-8 -*-
"""
GSE132188 official-method benchmark entry.

Runs the official comparison pipeline for all settings or one selected setting:
- scLineagetracer
- CellRank (official)
- WOT (official)
- CoSpar (official)
- GAN-based OT (official adapter, if configured)
"""

import argparse


OUT_ROOT = "/Users/wanghongye/python/scLineagetracer/classification/GSE132188/roc/benchmark_all_timepoints_official_plus"


def run_all(run_cospar: int):
    import compare_roc_official_alltime_remaining3 as alltime

    alltime.run_dataset("GSE132188", run_cospar=int(run_cospar))


def run_one(setting: str, run_cospar: int):
    import compare_roc_official_remaining3 as core
    import compare_roc_official_alltime_remaining3 as alltime

    metas = alltime.gen_meta_132188_all(OUT_ROOT)
    meta_map = {m["setting"]: m for m in metas}
    if setting not in meta_map:
        raise ValueError(f"Unknown setting: {setting}. Available: {sorted(meta_map.keys())}")
    core.run_one(meta_map[setting], run_cospar=int(run_cospar))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "one"], default="all")
    parser.add_argument(
        "--setting",
        choices=["UpTo_12.5", "UpTo_13.5", "UpTo_14.5", "All_15.5"],
        default="UpTo_12.5",
    )
    parser.add_argument("--run_cospar", type=int, default=1, help="1=run official CoSpar")
    args = parser.parse_args()

    if args.mode == "all":
        run_all(run_cospar=int(args.run_cospar))
    else:
        run_one(setting=str(args.setting), run_cospar=int(args.run_cospar))


if __name__ == "__main__":
    main()

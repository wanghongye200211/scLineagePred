# -*- coding: utf-8 -*-
"""
GSE140802 official-method comparison (single setting).

This script supersedes the old coarse/proxy version and runs only official
methods through the shared official pipeline:
- scLineagetracer
- CellRank
- WOT
- CoSpar
- GAN-based OT (official adapter, if available)

Outputs are written under:
- classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus/<SETTING>

Optionally triggers final 2D plots from official benchmark outputs.
"""

import os
import argparse


def run_one_setting(setting: str, out_root: str, run_cospar: int, make_2d: int):
    import pandas as pd
    import compare_roc_official_remaining3 as core
    import compare_roc_official_alltime_remaining3 as alltime

    core.ensure_dir(out_root)
    metas = alltime.gen_meta_140802_all(out_root)
    meta_map = {m["setting"]: m for m in metas}
    if setting not in meta_map:
        raise ValueError(f"Unknown setting: {setting}. Available: {sorted(meta_map.keys())}")

    meta = meta_map[setting]
    print(f"[RUN] GSE140802 {setting} (official methods)")
    core.run_one(meta, run_cospar=int(run_cospar))

    ms = os.path.join(meta["out_dir"], "metrics_summary.csv")
    if os.path.isfile(ms):
        df = pd.read_csv(ms)
        df.insert(0, "Dataset", "GSE140802")
        df.to_csv(os.path.join(meta["out_dir"], "metrics_summary_with_dataset.csv"), index=False)

    if int(make_2d) == 1:
        try:
            import plot_binary_official_methods_2d as plot2d
            plot2d.run_dataset("GSE140802")
            print("[DONE] 2D plots generated for GSE140802")
        except Exception as e:
            print(f"[WARN] 2D plotting skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", default="Day2_Only", choices=["Day2_Only", "Day2_Day4", "All_Days"])
    parser.add_argument(
        "--out_root",
        default="/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus",
    )
    parser.add_argument("--run_cospar", type=int, default=1, help="1=run official CoSpar")
    parser.add_argument("--make_2d", type=int, default=1, help="1=generate final 2D official plots")
    args = parser.parse_args()

    run_one_setting(
        setting=str(args.setting),
        out_root=str(args.out_root),
        run_cospar=int(args.run_cospar),
        make_2d=int(args.make_2d),
    )


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Day2 official-method comparison entry for GSE140802.

This script replaces the historical HVG coarse/proxy implementation and now
runs the official pipeline for Day2_Only.
"""

import argparse


def main():
    from compare_roc_cellrank_wot_cospar_140802 import run_one_setting

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_root",
        default="/Users/wanghongye/python/scLineagetracer/classification/GSE140802_Final_v7/roc_click/benchmark_all_timepoints_official_plus",
    )
    parser.add_argument("--run_cospar", type=int, default=1, help="1=run official CoSpar")
    parser.add_argument("--make_2d", type=int, default=1, help="1=generate final 2D official plots")
    args = parser.parse_args()

    run_one_setting(
        setting="Day2_Only",
        out_root=str(args.out_root),
        run_cospar=int(args.run_cospar),
        make_2d=int(args.make_2d),
    )


if __name__ == "__main__":
    main()

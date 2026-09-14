#!/usr/bin/env python3
"""Compare the test-only re-runs of §3/§4 against the previous test-only outputs.

Both runs use identical settings, so this isolates run-to-run variation: the
closed-form rows (Base / PCA / ZCA) should be bit-identical, while NCWP rows can
move within seed noise because GPU matmul reductions are not bit-reproducible.
Previous outputs were copied to rebuttal_outputs/_prev/ before re-running.
"""
import sys

import pandas as pd

PREV = "/workspace/NCWP/rebuttal_outputs/_prev"
OUT1 = "/workspace/NCWP/rebuttal_outputs"
OUT2 = "/workspace/NCWP/rebuttal_outputs_v2"

KEYS = ["model_nickname", "method", "r"]


def compare(name, prev_csv, new_csv, extra_keys=()):
    a = pd.read_csv(prev_csv)
    b = pd.read_csv(new_csv)
    keys = KEYS + [k for k in extra_keys if k in a.columns]
    m = a.merge(b, on=keys, suffixes=("_prev", "_new"))
    m["delta"] = m.metric_value_new - m.metric_value_prev

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  prev rows={len(a)}  new rows={len(b)}  matched={len(m)}")
    if len(m) != len(a) or len(m) != len(b):
        print("  ⚠ row sets differ — unmatched rows:")
        for df, tag in ((a, "prev"), (b, "new")):
            missing = df.merge(m[keys], on=keys, how="left", indicator=True)
            missing = missing[missing._merge == "left_only"]
            for _, r in missing.iterrows():
                print(f"      {tag}: {r.model_nickname} {r.method} r={r.r}")

    det = m[~m.method.str.contains("NCWP")]
    stoch = m[m.method.str.contains("NCWP")]
    print(f"\n  deterministic rows (Base/PCA/ZCA/Echo): n={len(det)} "
          f"max|delta|={det.delta.abs().max():.4f}")
    if det.delta.abs().max() > 1e-4:
        print("  ⚠ deterministic rows changed:")
        for _, r in det[det.delta.abs() > 1e-4].iterrows():
            print(f"      {r.model_nickname} {r.method} r={r.r}: "
                  f"{r.metric_value_prev:.4f} -> {r.metric_value_new:.4f}")
    else:
        print("  ✓ bit-identical")

    print(f"\n  NCWP rows: n={len(stoch)} mean delta={stoch.delta.mean():+.3f} "
          f"max|delta|={stoch.delta.abs().max():.3f}")
    over = stoch[stoch.delta.abs() > 1.0]
    print(f"  |delta| > 1.0: {len(over)} / {len(stoch)}")
    for _, r in stoch.reindex(stoch.delta.abs().sort_values(ascending=False).index).head(6).iterrows():
        sp = r.get("std_if_available_prev", float("nan"))
        sn = r.get("std_if_available_new", float("nan"))
        print(f"      {r.model_nickname:9s} {r.method:42s} r={r.r:>4} "
              f"{r.metric_value_prev:6.2f}±{sp:.2f} -> {r.metric_value_new:6.2f}±{sn:.2f} "
              f"({r.delta:+.2f})")
    return m


def main():
    ms = [
        compare("§3 Layer-selection, test-only 1,379",
                f"{PREV}/layer_testonly_prev.csv",
                f"{OUT2}/table_layer_selection_testonly1379.csv",
                extra_keys=("layer_name", "pooling")),
        compare("§4 Echo whitening, test-only 1,379",
                f"{PREV}/echo_testonly_prev.csv",
                f"{OUT1}/table_echo_whitening_testonly1379.csv"),
    ]
    all_ncwp = pd.concat([m[m.method.str.contains("NCWP")] for m in ms])
    print(f"\n{'=' * 78}\n종합\n{'=' * 78}")
    print(f"  NCWP 셀 {len(all_ncwp)}개, |delta| 최대 {all_ncwp.delta.abs().max():.2f}, "
          f"평균 {all_ncwp.delta.mean():+.3f}")
    print(f"  |delta| > 1.0 인 셀: {(all_ncwp.delta.abs() > 1.0).sum()}개")
    if (all_ncwp.delta.abs() > 1.5).any():
        print("  ⚠ 1.5 초과 셀 존재 — 확인 필요")
        sys.exit(1)
    print("  ✓ 모든 NCWP 변화가 1.5 이내 (seed noise 수준)")


if __name__ == "__main__":
    main()

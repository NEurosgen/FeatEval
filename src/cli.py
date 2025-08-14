# src/feateval/cli.py
import argparse, json
import pandas as pd
from feateval.data.splits import CVConfig, CVEngine
from feateval.models.sk_linear import SKLogReg
from feateval.evaluate.compare_sets import compare_sets
from feateval.selectors.filters import rank_by_mi
from feateval.selectors.search import sffs_beam_search
from feateval.selectors.stability import stability_selection

def main():
    ap = argparse.ArgumentParser("feval")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run-search")
    p.add_argument("--data", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--A", nargs="+", required=True, help="колонки A")
    p.add_argument("--B", nargs="+", required=True, help="колонки B")
    p.add_argument("--metric", default="pr_auc")
    p.add_argument("--cv", default="stratified")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topk", type=int, default=200)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--stab-thr", type=float, default=0.7)
    p.add_argument("--out", required=True)

    args = ap.parse_args()
    df = pd.read_parquet(args.data) if args.data.endswith((".parquet",".pq")) else pd.read_csv(args.data)

    cv_cfg = CVConfig(cv_type=args.cv, n_splits=args.folds, n_repeats=1, shuffle=True, seed=args.seed)
    cv = CVEngine(cv_cfg)

    res = compare_sets(df, args.target, args.A, args.B, cv, SKLogReg)
    start_group = "A" if (sum(res["A"]) / len(res["A"])) >= (sum(res["B"]) / len(res["B"])) else "B"
    S0 = args.A if start_group == "A" else args.B

    pool_all = sorted(set(args.A) | set(args.B))
    pool_cand = [c for c in pool_all if c not in S0]
    ranked = rank_by_mi(df[pool_all], df[args.target].to_numpy(), pool_cand)[:args.topk]

    S_mix, trace = sffs_beam_search(df, args.target, S0, ranked, cv, SKLogReg,
                                    beam=args.beam, patience=args.patience)

    def cv_factory(r):
        return CVEngine(CVConfig(cv_type=args.cv, n_splits=args.folds, n_repeats=1,
                                 shuffle=True, seed=args.seed + 17*r))

    S_star, freq = stability_selection(df, args.target, S0, ranked, cv_factory, SKLogReg,
                                       repeats=args.repeats, threshold=args.stab_thr,
                                       beam=args.beam, patience=args.patience)

    out = {
        "set_compare": res["diff"],
        "start_group": start_group,
        "mix_best": S_mix,
        "stability": {"subset": S_star, "freq": freq},
        "trace_top": trace[:200]
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()

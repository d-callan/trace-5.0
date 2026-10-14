#!/usr/bin/env python3
"""
Benchmark of DANNO and STEVE on the ProsperiLab/DYNAMITE nosoi transmission trees
from Rife Magalis et al. (2024), Infection, Genetics and Evolution 124:105661.

Evaluates:
1. Ground-truth patristic and temporal geometry across Static (B-E), Growing (F),
   and Declining (G) risk-group clusters embedded in Background (A) circulation.
2. Cluster recovery (Jaccard >= 0.50) and Background FPR comparing static HIV-TRACE
   distance thresholds against DANNO dynamic clock corridors and STEVE spectral
   modularity deconstruction.
3. Dynamic regime classification (discriminating Growing, Static, and Declining
   clusters) using sequence-only metrics vs. STEVE joint temporal-genetic metrics.
"""

import argparse
import glob
import os
import re
import sys
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from trace50.autoclock import optimize_fiedler_modularity


def parse_nexus_tree_fast(filepath):
    """Parse a nosoi NEXUS transmission tree into patristic distance matrix, states, and tip times."""
    with open(filepath) as f:
        text = f.read()
    t_match = re.search(r"TREE\s+.*?=\s*\[&R\]\s*(.*);", text, re.IGNORECASE)
    if not t_match:
        return None
    nwk = t_match.group(1)
    tip_meta = {}
    for m in re.finditer(r"(?<=[(,])(\d+)\[&([^\]]+)\]:([0-9.eE+-]+)", nwk):
        tid = m.group(1)
        kv = dict(item.split("=", 1) for item in m.group(2).split(",") if "=" in item)
        tip_meta[tid] = kv

    nwk_clean = re.sub(r"\[&[^\]]*\]", "", nwk)
    nwk_clean = re.sub(r"\)[A-Za-z0-9_.-]+", ")", nwk_clean)
    tokens = re.findall(r"\(|\)|,|:[0-9.eE+-]+|\d+", nwk_clean)
    parent_of = {}
    bl_of = {0: 0.0}
    curr = 0
    next_id = 1
    stack = []
    node_to_tid = {}
    for tok in tokens:
        if tok == "(":
            stack.append(curr)
            child = next_id
            next_id += 1
            parent_of[child] = curr
            bl_of[child] = 0.0
            curr = child
        elif tok == ",":
            curr = stack[-1]
            child = next_id
            next_id += 1
            parent_of[child] = curr
            bl_of[child] = 0.0
            curr = child
        elif tok == ")":
            curr = stack.pop()
        elif tok.startswith(":"):
            bl_of[curr] = float(tok[1:])
        else:
            node_to_tid[curr] = tok

    d_root = {0: 0.0}
    for u in range(1, next_id):
        p = parent_of[u]
        d_root[u] = d_root[p] + bl_of[u]

    tips = sorted(node_to_tid.keys())
    n_tips = len(tips)
    children_of = {u: [] for u in range(next_id)}
    for u, p in parent_of.items():
        children_of[p].append(u)
    tip_idx_of = {u: i for i, u in enumerate(tips)}
    desc_tips = {u: [] for u in range(next_id)}
    for u in range(next_id - 1, -1, -1):
        if u in tip_idx_of:
            desc_tips[u].append(tip_idx_of[u])
        for c in children_of[u]:
            desc_tips[u].extend(desc_tips[c])

    lca_depth = np.zeros((n_tips, n_tips))
    for u in range(next_id):
        ch = children_of[u]
        if len(ch) >= 2:
            du = d_root[u]
            for i_c in range(len(ch)):
                t1 = desc_tips[ch[i_c]]
                for j_c in range(i_c + 1, len(ch)):
                    t2 = desc_tips[ch[j_c]]
                    lca_depth[np.ix_(t1, t2)] = du
                    lca_depth[np.ix_(t2, t1)] = du

    tip_d_root = np.array([d_root[u] for u in tips])
    dist_mat = tip_d_root[:, None] + tip_d_root[None, :] - 2.0 * lca_depth
    np.fill_diagonal(dist_mat, 0.0)

    states = []
    times = []
    for u in tips:
        tid = node_to_tid[u]
        meta = tip_meta.get(tid, {})
        states.append(meta.get("state", "?"))
        times.append(float(meta.get("time", 0.0)))
    return dist_mat, np.array(states), np.array(times)


def eval_clusters(labels, states, min_size=2):
    res_strict = {"Static": [], "Growing": [], "Declining": []}
    for st, cat in [
        ("B", "Static"),
        ("C", "Static"),
        ("D", "Static"),
        ("E", "Static"),
        ("F", "Growing"),
        ("G", "Declining"),
    ]:
        idx = np.where(states == st)[0]
        if len(idx) < 4:
            continue
        cl_labels = labels[idx]
        valid_c = [c for c in set(cl_labels) if c >= 0]
        best_j = 0.0
        for c in valid_c:
            c_members = np.where(labels == c)[0]
            if len(c_members) < min_size:
                continue
            inter = len(np.intersect1d(idx, c_members))
            union = len(np.union1d(idx, c_members))
            best_j = max(best_j, inter / union)
        res_strict[cat].append(best_j >= 0.50)
    idx_A = np.where(states == "A")[0]
    clustered_A = 0
    for u in idx_A:
        c = labels[u]
        if c >= 0 and np.sum(labels == c) >= min_size:
            clustered_A += 1
    fpr_A = clustered_A / len(idx_A) if len(idx_A) > 0 else 0.0
    return res_strict, fpr_A


def recursive_steve_split(K_mat, adj_mat, comp, max_cl_size=24, min_q=0.15):
    final_labels = comp.copy()
    next_label = int(np.max(comp)) + 1
    queue = [c_id for c_id in range(next_label)]
    while queue:
        c_id = queue.pop(0)
        members = np.where(final_labels == c_id)[0]
        if len(members) > max_cl_size:
            sub_K = K_mat[np.ix_(members, members)]
            sub_adj = adj_mat[np.ix_(members, members)].astype(float)
            pos_k = sub_K[sub_adj > 0]
            sigma = max(np.median(pos_k) if len(pos_k) > 0 else 1.0, 0.5)
            W = sub_adj * np.exp(-(sub_K ** 2) / (2 * sigma ** 2))
            deg = W.sum(axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
            L_sym = np.eye(len(members)) - D_inv_sqrt @ W @ D_inv_sqrt
            evals, evecs = np.linalg.eigh(L_sym)
            if len(evals) > 1:
                fiedler = D_inv_sqrt @ evecs[:, 1]
                c0, c1, best_q, _ = optimize_fiedler_modularity(W, fiedler)
                if best_q > min_q and len(c0) >= 3 and len(c1) >= 3:
                    final_labels[members[c1]] = next_label
                    queue.append(c_id)
                    queue.append(next_label)
                    next_label += 1
    return final_labels


def calc_auc(pos_scores, neg_scores):
    u_stat, p_val = mannwhitneyu(pos_scores, neg_scores, alternative="two-sided")
    auc = u_stat / (len(pos_scores) * len(neg_scores))
    return max(auc, 1.0 - auc), p_val


def main():
    parser = argparse.ArgumentParser(description="Run DYNAMITE nosoi benchmark.")
    parser.add_argument(
        "--trees-dir",
        type=str,
        required=True,
        help="Directory containing sim_*_sampled_10000.tree files from ProsperiLab/DYNAMITE",
    )
    parser.add_argument("--max-trees", type=int, default=150)
    args = parser.parse_args()

    tree_files = sorted(glob.glob(os.path.join(args.trees_dir, "*.tree")))[: args.max_trees]
    rng = np.random.default_rng(42)
    mu_L = 0.8219  # nosoi_sim.R exact sequence parameterization (L=1000, mu=8.219e-4/day)

    methods = {
        "HIV-TRACE (1.0%, K<=10)": {"Static": [], "Growing": [], "Declining": [], "FPR": []},
        "DANNO (d0=1.0%+mu*dt)": {"Static": [], "Growing": [], "Declining": [], "FPR": []},
        "HIV-TRACE (1.5%, K<=15)": {"Static": [], "Growing": [], "Declining": [], "FPR": []},
        "DANNO+STEVE (K<=13+mu*L*dt)": {"Static": [], "Growing": [], "Declining": [], "FPR": []},
    }
    records = []

    for tf in tree_files:
        parsed = parse_nexus_tree_fast(tf)
        if parsed is None:
            continue
        d_pat, states, times = parsed
        n = len(states)
        iu, ju = np.triu_indices(n, k=1)
        k_vals = rng.poisson(mu_L * d_pat[iu, ju])
        K_mat = np.zeros((n, n), dtype=float)
        K_mat[iu, ju] = k_vals
        K_mat[ju, iu] = k_vals
        dt_mat = np.abs(times[:, None] - times[None, :])
        t_max = np.max(times)

        for name, kc in [("HIV-TRACE (1.0%, K<=10)", 10), ("HIV-TRACE (1.5%, K<=15)", 15)]:
            adj = (K_mat <= kc).astype(int)
            np.fill_diagonal(adj, 0)
            _, comp = connected_components(csr_matrix(adj), directed=False)
            r_s, fpr = eval_clusters(comp, states)
            for cat in ["Static", "Growing", "Declining"]:
                methods[name][cat].extend(r_s[cat])
            methods[name]["FPR"].append(fpr)

        adj1 = (K_mat <= (10.0 + mu_L * dt_mat)) & (dt_mat <= 14.0)
        np.fill_diagonal(adj1, False)
        _, comp1 = connected_components(csr_matrix(adj1.astype(int)), directed=False)
        r_s, fpr = eval_clusters(comp1, states)
        for cat in ["Static", "Growing", "Declining"]:
            methods["DANNO (d0=1.0%+mu*dt)"][cat].extend(r_s[cat])
        methods["DANNO (d0=1.0%+mu*dt)"]["FPR"].append(fpr)

        adj2 = (K_mat <= (13.0 + mu_L * dt_mat)) & (dt_mat <= 14.0)
        np.fill_diagonal(adj2, False)
        _, comp2 = connected_components(csr_matrix(adj2.astype(int)), directed=False)
        comp2_s = recursive_steve_split(K_mat, adj2, comp2, max_cl_size=24, min_q=0.15)
        r_s, fpr = eval_clusters(comp2_s, states)
        for cat in ["Static", "Growing", "Declining"]:
            methods["DANNO+STEVE (K<=13+mu*L*dt)"][cat].extend(r_s[cat])
        methods["DANNO+STEVE (K<=13+mu*L*dt)"]["FPR"].append(fpr)

        for st, cat in [
            ("B", "Static"),
            ("C", "Static"),
            ("D", "Static"),
            ("E", "Static"),
            ("F", "Growing"),
            ("G", "Declining"),
        ]:
            idx = np.where(states == st)[0]
            if len(idx) < 4:
                continue
            sub_K = K_mat[np.ix_(idx, idx)].copy()
            sub_t = times[idx]
            iu_c, ju_c = np.triu_indices(len(idx), k=1)
            mean_k = np.mean(sub_K[iu_c, ju_c])
            np.fill_diagonal(sub_K, np.inf)
            min_k = np.min(sub_K, axis=1)
            med_nn_k = np.median(min_k)
            mean_nn_k = np.mean(min_k)
            t_span = np.max(sub_t) - np.min(sub_t)
            mean_dt = np.mean(np.abs(sub_t[:, None] - sub_t[None, :])[iu_c, ju_c])
            mean_age = np.mean(t_max - sub_t)
            rec_rate = np.sum((t_max - sub_t) <= 3.0) / max(len(idx), 1)
            records.append(
                {
                    "cat": cat,
                    "size": len(idx),
                    "mean_k": mean_k,
                    "med_nn_k": med_nn_k,
                    "mean_nn_k": mean_nn_k,
                    "t_span": t_span,
                    "mean_dt": mean_dt,
                    "mean_age": mean_age,
                    "steve_declining_score": mean_nn_k * (mean_age + 0.5) / (rec_rate + 0.1),
                    "steve_dynamic_ratio": (mean_dt + 0.1) / (mean_k + 1.0),
                }
            )

    print("=== Cluster Recovery Benchmark (150 nosoi trees, 767 clusters) ===")
    for m, d in methods.items():
        st_tpr = np.mean(d["Static"]) * 100
        gr_tpr = np.mean(d["Growing"]) * 100
        dc_tpr = np.mean(d["Declining"]) * 100
        macro_tpr = (st_tpr + gr_tpr + dc_tpr) / 3.0
        fpr_m = np.mean(d["FPR"]) * 100
        bal_acc = 0.5 * (macro_tpr + (100.0 - fpr_m))
        print(
            f"  {m:28s}: Static={st_tpr:5.1f}%, Growing={gr_tpr:5.1f}%, Declining={dc_tpr:5.1f}% "
            f"| MacroTPR={macro_tpr:5.1f}% | BgFPR={fpr_m:5.1f}% | BalAcc={bal_acc:5.1f}%"
        )

    st_rec = [r for r in records if r["cat"] == "Static"]
    gr_rec = [r for r in records if r["cat"] == "Growing"]
    dc_rec = [r for r in records if r["cat"] == "Declining"]

    print("\n=== Regime Discrimination (AUC & Mann-Whitney p-value) ===")
    for feat in ["mean_k", "med_nn_k", "t_span", "mean_dt", "mean_age", "steve_declining_score"]:
        auc1, p1 = calc_auc([r[feat] for r in dc_rec], [r[feat] for r in st_rec])
        print(f"  Declining vs Static ({feat:22s}): AUC={auc1:.3f} (p={p1:.2e})")
    for feat in ["mean_k", "med_nn_k", "mean_dt", "t_span", "steve_dynamic_ratio"]:
        auc, p = calc_auc([r[feat] for r in gr_rec], [r[feat] for r in st_rec])
        print(f"  Growing vs Static   ({feat:22s}): AUC={auc:.3f} (p={p:.2e})")


if __name__ == "__main__":
    main()

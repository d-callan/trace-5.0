"""Oracle for tests/test_seq_len_oracle.rs (C3, effective sequence length L).

L is computed exactly as validation/run_danno.py does (lines 249-260): per raw
FASTA row, count characters != '-', take numpy.median, truncate with int().
The Bayes factors come from the Python reference
trace50.bayes_factor.DannoEstimator(seq_len=L, ...) with the same parameters
the Rust test passes to DannoEstimator::new.

Run from the repo root:  python3 trace50-rust/crates/trace50-core/tests/fixtures/seq_len/gen_oracle.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *(['..'] * 6)))
sys.path.insert(0, REPO)
from trace50.bayes_factor import DannoEstimator  # noqa: E402

msa = os.path.join(HERE, 'ragged_toy.fasta')
# --- verbatim logic of validation/run_danno.py
msa_width, nongap = None, []
with open(msa) as fh:
    for line in fh:
        if not line.startswith('>'):
            s = line.strip()
            if msa_width is None:
                msa_width = len(s)
            nongap.append(len(s) - s.count('-'))
eff_len = int(np.median(nongap)) if nongap else msa_width
# ---
width = max(len(l.strip()) for l in open(msa) if not l.startswith('>'))

params = dict(mu=2.0e-3, tau_bar=1.0, omega=2.0, t_span=20.0, alpha_adequacy=0.05,
              bg_mean_dist=0.065, bg_var_dist=0.0004)
est = DannoEstimator(seq_len=eff_len, **params)
cases = []
for k, dt in [(0, 0.0), (0, 0.5), (1, 0.25), (1, 1.5), (2, 1.0), (3, 2.0)]:
    d = k / eff_len
    cases.append({'dist': d, 'delta_t': dt, 'bf': float(est.compute_bayes_factor(d, dt))})
for d, dt in [(0.01, 0.3), (0.05, 1.0)]:
    cases.append({'dist': d, 'delta_t': dt, 'bf': float(est.compute_bayes_factor(d, dt))})

out = {'nongap_counts': nongap, 'median_nongap_numpy': float(np.median(nongap)),
       'effective_len': eff_len, 'alignment_width': width, 'params': params,
       'p_trans': float(est.p_trans), 'p_bg': float(est.p_bg), 'r_bg': float(est.r_bg),
       'cases': cases}
json.dump(out, open(os.path.join(HERE, 'oracle.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))

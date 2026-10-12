//! TN93 parity against the `tn93` binary (v1.1.0, `tn93 -t 1.0 -a RMYSKB`).
//!
//! Fixtures and the scripts that produced them live in `tests/fixtures/tn93/`;
//! every expected value here is read from a CSV written by the tn93 binary.

use std::collections::BTreeMap;
use trace50_core::Alignment;

/// Parses tn93 CSV output (`ID1,ID2,Distance`) into an unordered-pair map.
fn parse_tn93_csv(csv: &str) -> BTreeMap<(String, String), f64> {
    let mut out = BTreeMap::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').collect();
        let (a, b) = (f[0].to_string(), f[1].to_string());
        let key = if a <= b { (a, b) } else { (b, a) };
        out.insert(key, f[2].trim().parse::<f64>().unwrap());
    }
    out
}

/// Pairs with a finite internal distance, keyed like `parse_tn93_csv`.
fn rust_pairs(aln: &Alignment) -> BTreeMap<(String, String), f64> {
    let n = aln.num_sequences();
    let m = aln.compute_distance_matrix();
    let mut out = BTreeMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = m[i * n + j];
            if d.is_finite() {
                let (a, b) = (aln.headers[i].clone(), aln.headers[j].clone());
                let key = if a <= b { (a, b) } else { (b, a) };
                out.insert(key, d);
            }
        }
    }
    out
}

/// tn93 scores a pair only when it shares more than 100 non-gap sites. The toy
/// alignment straddles that boundary at 100 vs 101 shared sites, including a
/// gapped (every other column) sequence and one whose 101st site is an N.
#[test]
fn test_min_overlap_pair_set_matches_tn93_binary() {
    let aln = Alignment::from_fasta_str(include_str!("fixtures/tn93/overlap_toy.fasta")).unwrap();
    let oracle = parse_tn93_csv(include_str!("fixtures/tn93/overlap_toy_RMYSKB.csv"));
    let got = rust_pairs(&aln);

    let oracle_keys: Vec<_> = oracle.keys().cloned().collect();
    let got_keys: Vec<_> = got.keys().cloned().collect();
    assert_eq!(got_keys, oracle_keys, "pair set differs from tn93 -l 100");
    assert_eq!(oracle.len(), 4);
}

/// Pairs without a TN93 distance must stay out of every downstream stage
/// (static graph, candidate screen, AutoClock) rather than leak in as 0 or NaN.
/// Two groups occupy disjoint halves of a 300-column alignment, so no
/// cross-group pair has any shared site.
#[test]
fn test_unscorable_pairs_are_missing_downstream() {
    use trace50_core::{run_trace50, Trace50Config};

    let half = |seed: u64| -> String {
        // small deterministic LCG; content is irrelevant beyond being ACGT
        let mut x = seed;
        (0..150)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                b"ACGT"[(x >> 62) as usize] as char
            })
            .collect()
    };
    let (left, right) = (half(1), half(2));
    let gaps = "-".repeat(150);
    let mut fasta = String::new();
    for k in 0..4 {
        let mut l = left.clone().into_bytes();
        let mut r = right.clone().into_bytes();
        l[10 * k] = if l[10 * k] == b'A' { b'G' } else { b'A' };
        r[10 * k] = if r[10 * k] == b'C' { b'T' } else { b'C' };
        fasta.push_str(&format!(">L{}_{}\n{}{}\n", k, 2015 + k, String::from_utf8(l).unwrap(), gaps));
        fasta.push_str(&format!(">R{}_{}\n{}{}\n", k, 2015 + k, gaps, String::from_utf8(r).unwrap()));
    }
    let aln = Alignment::from_fasta_str(&fasta).unwrap();
    let n = aln.num_sequences();
    let m = aln.compute_distance_matrix();
    for i in 0..n {
        for j in (i + 1)..n {
            let cross = aln.headers[i].as_bytes()[0] != aln.headers[j].as_bytes()[0];
            assert_eq!(m[i * n + j].is_nan(), cross, "{} vs {}", aln.headers[i], aln.headers[j]);
        }
    }

    let mut cfg = Trace50Config::default();
    cfg.min_cluster_size = 2;
    cfg.static_distance_threshold = 0.05;
    let dossier = run_trace50(&aln, &cfg).unwrap();
    for e in &dossier.edges {
        assert_eq!(e.source.as_bytes()[0], e.target.as_bytes()[0], "cross-group edge {:?}", e.source);
        assert!(e.distance.is_finite());
    }
    for c in &dossier.communities {
        assert!(c.n_eff.is_finite() && c.mu.is_finite() && c.r_squared.is_finite(), "{:?}", c.path);
    }
}

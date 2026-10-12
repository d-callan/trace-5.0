//! CLI scorecard with zero supported edges (fixture: trace50-core/tests/fixtures/zero_edge).
//! N_act must read "not estimable", the reduction "n/a", and no inf/NaN or fake
//! percentage may be printed.

use std::path::PathBuf;
use std::process::Command;

fn run_cli(fixture: &str, out_name: &str) -> String {
    let fasta = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../trace50-core/tests/fixtures/zero_edge")
        .join(fixture);
    let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(out_name);
    let output = Command::new(env!("CARGO_BIN_EXE_trace50"))
        .arg("-f").arg(&fasta)
        .arg("-o").arg(&out_dir)
        .output()
        .expect("trace50 runs");
    assert!(output.status.success(), "trace50 failed: {}", String::from_utf8_lossy(&output.stderr));
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}

#[test]
fn test_cli_zero_edge_scorecard() {
    let text = run_cli("static_only.fasta", "zero_edge_cli");
    let lower = text.to_lowercase();
    assert!(!lower.contains("inf"), "scorecard prints inf:\n{}", text);
    assert!(!lower.contains("nan"), "scorecard prints NaN:\n{}", text);
    assert!(!text.contains("99.9%") && !text.contains("100.0%"), "fake reduction:\n{}", text);
    assert!(text.contains("Active Transmitting Pool (N_act): not estimable: zero supported edges (E_obs = 0)"), "{}", text);
    assert!(text.contains("Giant Component Reduction:        n/a"), "{}", text);
    // DANNO giant component is 0, not a singleton.
    let danno = text.split("DANNO Calibrated Transmission Network:").nth(1).unwrap();
    assert!(danno.contains("Giant Component Size:             0 patients"), "{}", text);
}

#[test]
fn test_cli_control_scorecard_reports_n_act() {
    // Control: 3 supported edges (run_danno oracle); N_act = 9 [2.9 - 27.9] (Python ChinEstimator).
    let text = run_cli("control.fasta", "zero_edge_cli_control");
    assert!(text.contains("Active Transmitting Pool (N_act): 9 [CI: 3 - 28]"), "{}", text);
    assert!(text.contains("Giant Component Reduction:        0.0%"), "{}", text);
}

//! Tier-1 tests for the fit-end check that an explicit ODE stepper did not inflate the analytic
//! `H` (#2846).
//!
//! An explicit stepper's error estimate reads values only. A fast mode that the predictions never
//! excite but the η-tangents do lets it step past its stability limit: the predictions stay
//! clean, the analytic Jacobian blows up, and the FOCEI determinant term (hence the reported OFV)
//! with it. The check re-scores each subject with a finite-difference `H` and warns when the
//! objective moves by `EXPLICIT_H_OFV_TOLERANCE` or more. These tests pin the arithmetic, every
//! sentence of the message, the gates, and the end-to-end wiring on the defect's own fixture.
use super::*;
use crate::api::postfit::{
    explicit_h_check, explicit_h_check_warning, explicit_h_summary, ExplicitHCheck,
    EXPLICIT_H_OFV_TOLERANCE,
};
use crate::parser::model_parser::parse_model_string;
use crate::types::{classify_warning, WarningCode, WarningSeverity};
use std::collections::HashMap;

/// Endosomal recycling with a binding quadratic: the fast pool (rate ≈ 20 /h) sits on its slow
/// manifold in the primal solution, so the error estimate never sees it, while the η-tangents
/// (through `Q`) excite it at O(1). `{METHOD}` is the `ode_method`.
fn fast_pool_model(method: &str) -> CompiledModel {
    let src = r#"
[parameters]
  theta TVCL(0.15, 0.001, 100)
  theta TVV1(3.0, 0.1, 100)
  theta TVQ(0.5, 0.001, 100)
  theta TVV2(4.0, 0.1, 100)
  theta TVCLUP(0.4, 0.001, 100)
  theta TVKREC(20, 0.0001, 100)
  theta TVKDEGE(100, 0.001, 1000)
  omega ETA_Q ~ 0.25
  sigma PROP_ERR ~ 0.1 (sd)
  sigma ADD_ERR ~ 0.0035 (sd)
[individual_parameters]
  CL = TVCL
  V1 = TVV1
  Q = TVQ * exp(ETA_Q)
  V2 = TVV2
  CLUP = TVCLUP
  KREC = TVKREC
  KDEGE = TVKDEGE
[structural_model]
  ode(states=[A1, A2, E])
[odes]
  C1 = A1 / V1
  C2 = A2 / V2
  X = E / 0.02
  B = X - 4e7 - 3e4
  CE = 0.5 * (B + sqrt(B * B + 4 * 3e4 * X))
  REC = KREC * (E - CE * 0.02)
  d/dt(A1) = -CL * C1 - Q * (C1 - C2) - CLUP * C1 + REC
  d/dt(A2) = Q * (C1 - C2)
  d/dt(E)  = CLUP * C1 - REC - KDEGE * CE * 0.02
[scaling]
  y = A1 / V1
[error_model]
  DV ~ combined(PROP_ERR, ADD_ERR)
[fit_options]
  method  = focei
  ode_method = METHOD
  ode_reltol = 1e-6
  ode_abstol = 1e-9
  ode_max_steps = 1000000
"#
    .replace("METHOD", method);
    parse_model_string(&src).expect("parse")
}

/// Subjects 1 and 4 of the #2846 repro (a 24-subject synthetic study drawn from this model), 300 mg
/// IV bolus. Real draws keep every η̂ moderate; a dataset the model fits badly pushes the
/// finite-difference reference itself off by more than a unit of OFV (measured while writing this).
const OBS_TIMES: [f64; 10] = [0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 48.0, 96.0, 168.0, 336.0];
const OBS: [[f64; 10]; 2] = [
    [
        94.8432, 79.7048, 67.8917, 54.1823, 36.2972, 21.5589, 14.3771, 5.46789, 1.06026, 0.048079,
    ],
    [
        83.7775, 97.9102, 65.102, 68.6879, 32.8083, 19.9622, 10.6688, 5.61985, 1.42196, 0.0804764,
    ],
];

fn subject(id: &str, obs: &[f64; 10]) -> Subject {
    Subject {
        obs_rgroup: Vec::new(),
        id: id.into(),
        doses: vec![DoseEvent::new(0.0, 300.0, 1, 0.0, false, 0.0)],
        obs_times: OBS_TIMES.to_vec(),
        obs_raw_times: Vec::new(),
        observations: obs.to_vec(),
        obs_cmts: vec![1; 10],
        covariates: HashMap::new(),
        dose_covariates: Vec::new(),
        obs_covariates: Vec::new(),
        pk_only_times: Vec::new(),
        pk_only_raw_times: Vec::new(),
        reset_period_times: Vec::new(),
        pk_only_covariates: Vec::new(),
        reset_times: Vec::new(),
        reset_covariates: Vec::new(),
        cmt_toggles: Vec::new(),
        cens: vec![0; 10],
        occasions: Vec::new(),
        obs_l2: Vec::new(),
        dose_occasions: Vec::new(),
        reset_occasions: Vec::new(),
        pk_only_occasions: Vec::new(),
        fremtype: Vec::new(),
        obs_records: vec![],
        endpoint_covariates: Vec::new(),
        filled_cells: Vec::new(),
        simulated_mixnum: None,
    }
}

fn population() -> Population {
    Population {
        subjects: OBS
            .iter()
            .enumerate()
            .map(|(i, o)| subject(&(i + 1).to_string(), o))
            .collect(),
        covariate_names: Vec::new(),
        dv_column: "DV".into(),
        input_columns: Vec::new(),
        exclusions: None,
        warnings: Vec::new(),
    }
}

/// An evaluation at the initial estimates (`outer_maxiter = 0`), which is what the repro runs.
fn fit_with(model: &CompiledModel) -> FitResult {
    let options = FitOptions {
        outer_maxiter: 0,
        run_covariance_step: false,
        interaction: true,
        ode_reltol: 1e-6,
        ode_abstol: 1e-9,
        ode_max_steps: 1_000_000,
        ode_method: model
            .ode_spec
            .as_ref()
            .expect("an ODE model")
            .solver_opts
            .method,
        ..FitOptions::default()
    };
    crate::fit(
        model,
        &population(),
        &model.default_params.clone(),
        &options,
    )
    .expect("fit runs")
}

/// What the fit hands the check: the EBEs and analytic `H` of the final estimates (the fit ran
/// zero outer iterations, so those are the initial estimates) and the reported per-subject OFV.
struct Finals {
    params: ModelParameters,
    etas: Vec<DVector<f64>>,
    subjects: Vec<SubjectResult>,
}

fn finals(model: &CompiledModel, result: &FitResult) -> Finals {
    let params = model.default_params.clone();
    let etas: Vec<DVector<f64>> = result.subjects.iter().map(|s| s.eta.clone()).collect();
    Finals {
        params,
        etas,
        subjects: result.subjects.clone(),
    }
}

fn check_of(model: &CompiledModel, result: &FitResult, mixture: bool) -> Option<ExplicitHCheck> {
    let f = finals(model, result);
    explicit_h_check(
        model,
        &population(),
        &f.params,
        &f.etas,
        &f.subjects,
        true,
        mixture,
    )
}

fn explicit_h_warning(result: &FitResult) -> Option<&String> {
    result
        .warnings
        .iter()
        .find(|w| w.starts_with("W_ODE_EXPLICIT_UNSTABLE_TANGENTS:"))
}

// ── arithmetic ───────────────────────────────────────────────────────────────

fn ids(n: usize) -> Vec<Subject> {
    (0..n).map(|i| subject(&format!("s{i}"), &OBS[0])).collect()
}

#[test]
fn the_summary_counts_totals_and_names_the_worst_subject() {
    let subjects = ids(5);
    let deltas = [Some(0.4), Some(-3.0), None, Some(2.5), Some(-1.0)];
    for d in deltas.iter().flatten() {
        assert!(f64::is_finite(*d));
    }
    let c = explicit_h_summary(&subjects, &deltas).expect("four subjects were scored");
    assert_eq!(c.checked, 4, "the None entry is skipped, not counted");
    assert_eq!(
        c.exceeding, 3,
        "|-3.0|, |2.5| and |-1.0| are at or above 1.0"
    );
    assert!(
        (c.delta_ofv - (0.4 - 3.0 + 2.5 - 1.0)).abs() < 1e-12,
        "{c:?}"
    );
    assert_eq!(c.worst_id, "s1", "the largest |delta| wins, sign kept");
    assert_eq!(c.worst_delta, -3.0);
}

#[test]
fn the_summary_is_none_when_no_subject_was_scored() {
    assert_eq!(explicit_h_summary(&ids(2), &[None, None]), None);
    assert_eq!(explicit_h_summary(&[], &[]), None);
}

/// The tolerance is inclusive: a subject at exactly the tolerance counts, one ulp below does not.
#[test]
fn the_tolerance_is_inclusive_and_sign_blind() {
    let t = EXPLICIT_H_OFV_TOLERANCE;
    assert_eq!(t, 1.0, "the documented threshold");
    let below = f64::from_bits(t.to_bits() - 1);
    let s = ids(1);
    assert_eq!(explicit_h_summary(&s, &[Some(t)]).unwrap().exceeding, 1);
    assert_eq!(explicit_h_summary(&s, &[Some(-t)]).unwrap().exceeding, 1);
    assert_eq!(explicit_h_summary(&s, &[Some(below)]).unwrap().exceeding, 0);
    assert_eq!(
        explicit_h_summary(&s, &[Some(-below)]).unwrap().exceeding,
        0
    );
}

// ── message ──────────────────────────────────────────────────────────────────

fn flagged() -> ExplicitHCheck {
    ExplicitHCheck {
        checked: 24,
        exceeding: 23,
        delta_ofv: -1088.6126,
        worst_id: "S23".into(),
        worst_delta: -90.0551,
    }
}

#[test]
fn no_warning_when_no_subject_exceeds() {
    let quiet = ExplicitHCheck {
        exceeding: 0,
        ..flagged()
    };
    assert!(explicit_h_check_warning(&quiet).is_none());
    assert!(explicit_h_check_warning(&flagged()).is_some());
}

/// Each sentence of the message, with the negative that a wrong message would satisfy: the
/// counts, the worst subject and the total come from the check, and the advice names only what
/// was measured to cure the repro.
#[test]
fn every_sentence_of_the_message_says_what_it_should() {
    let (msg, entry) = explicit_h_check_warning(&flagged()).unwrap();
    assert_eq!(entry.message, msg);
    for (sentence, why) in [
        (
            "W_ODE_EXPLICIT_UNSTABLE_TANGENTS: the objective of 23 of 24 subjects moves by 1 or more",
            "the token, and the exceeding/checked counts",
        ),
        (
            "when the Jacobian of the predictions is rebuilt from finite differences at the final estimates",
            "what the reference is",
        ),
        ("subject S23 moves by -90.0551", "the worst subject and its signed delta"),
        ("all subjects by -1088.6126 together", "the signed total"),
        (
            "An explicit ode_method (rk45, vern7, or auto where it stayed explicit) sets its step from the predictions alone",
            "the cause: which methods, and why the step is blind",
        ),
        (
            "step past the stability limit of a fast mode that only the eta-sensitivities excite",
            "the mechanism",
        ),
        (
            "the predictions stay clean while the analytic Jacobian, and with it the determinant term of the FOCEI objective, is inflated",
            "what the user sees and what is wrong",
        ),
        ("Treat the reported OFV as suspect", "the consequence"),
        (
            "set ode_method = rodas5p or tighten ode_reltol, and refit",
            "the measured fix",
        ),
        ("(#2846)", "the issue reference"),
    ] {
        assert!(msg.contains(sentence), "missing ({why}): {sentence:?}\n{msg}");
    }
    // The negatives: no advice that was not measured to work, no wrong subject or sign.
    for wrong in [
        "ode_auto_switch",
        "subject S22",
        "by +90.0551",
        "by +1088.6126",
    ] {
        assert!(!msg.contains(wrong), "unexpected {wrong:?} in {msg}");
    }
    let d = entry.details.as_ref().expect("details payload");
    assert_eq!(d["subjects_checked"], 24);
    assert_eq!(d["subjects_exceeding"], 23);
    assert_eq!(d["tolerance"], EXPLICIT_H_OFV_TOLERANCE);
    assert_eq!(d["worst_subject"], "S23");
    assert_eq!(d["worst_delta_ofv"], -90.0551);
    assert_eq!(d["total_delta_ofv"], -1088.6126);
}

#[test]
fn the_token_classifies_as_an_ode_solver_warning() {
    let (msg, entry) = explicit_h_check_warning(&flagged()).unwrap();
    let c = classify_warning(&msg);
    assert_eq!(c.category, WarningCode::OdeSolver);
    assert_eq!(c.severity, WarningSeverity::Warning);
    assert_eq!(entry.category, WarningCode::OdeSolver);
    assert_eq!(entry.severity, WarningSeverity::Warning);
}

// ── the defect, end to end ───────────────────────────────────────────────────

/// On the defect's own fixture the explicit steppers step past the stability limit, and the check
/// sees it in both warning channels; the stiff method (`rodas5p`) and the one explicit method
/// whose step stays clear (`vern7`) are silent. The realised numbers are printed.
#[test]
fn rk45_on_a_tangent_only_fast_mode_warns_and_a_stiff_method_does_not() {
    let rk45 = fast_pool_model("rk45");
    let r = fit_with(&rk45);
    let check = check_of(&rk45, &r, false).expect("rk45 on the analytic path is checked");
    eprintln!("rk45: {check:?}");
    assert!(check.exceeding >= 1, "{check:?}");
    // Measured (macOS, ci-fast): the worst subject moves by -12.8 and the total by -12.7; at the
    // repro's own tolerances (1e-8 / 1e-11) 79.6 and 178.1. 5x the threshold keeps 2.5x headroom.
    assert!(
        check.worst_delta < -5.0 * EXPLICIT_H_OFV_TOLERANCE,
        "{check:?}"
    );
    assert!(
        check.delta_ofv < 0.0,
        "the inflated analytic H lowers OFV when replaced: {check:?}"
    );
    let msg = explicit_h_warning(&r).unwrap_or_else(|| panic!("no warning in {:?}", r.warnings));
    assert!(msg.contains("(#2846)"));
    assert!(
        msg.contains(" moves by -"),
        "the signed worst delta reaches the message: {msg}"
    );
    // The typed channel carries the check's own entry (with its details payload), not just a
    // string the classifier re-derives.
    assert!(
        r.warnings_structured
            .iter()
            .any(|e| e.category == WarningCode::OdeSolver
                && e.message == *msg
                && e.details
                    .as_ref()
                    .is_some_and(|d| d["subjects_exceeding"].as_u64() >= Some(1))),
        "the typed channel carries the same warning with its details: {:?}",
        r.warnings_structured
    );

    for method in ["rodas5p", "vern7"] {
        let model = fast_pool_model(method);
        let r = fit_with(&model);
        let check = check_of(&model, &r, false);
        eprintln!("{method}: {check:?}");
        assert!(
            explicit_h_warning(&r).is_none(),
            "{method}: {:?}",
            r.warnings
        );
        if method == "vern7" {
            // Checked (vern7 is explicit) and clean: the realised worst is ~7e-6, 5 orders under
            // the threshold.
            let c = check.expect("vern7 is explicit, so it is checked");
            assert!(f64::is_finite(c.worst_delta));
            assert!(
                c.worst_delta.abs() < 1e-2 * EXPLICIT_H_OFV_TOLERANCE,
                "{c:?}"
            );
        } else {
            assert!(check.is_none(), "rodas5p is never explicit");
        }
    }
}

// ── gates ────────────────────────────────────────────────────────────────────

/// One fit, then each condition that makes the check decline flipped in turn from a control that
/// is checked: the control proves the others decline for their own reason, not because the
/// inputs were never checkable.
#[test]
fn each_gate_declines_on_its_own() {
    let model = fast_pool_model("rk45");
    let r = fit_with(&model);
    let f = finals(&model, &r);
    let run = |m: &CompiledModel, etas: &[DVector<f64>], mixture: bool| {
        explicit_h_check(
            m,
            &population(),
            &f.params,
            etas,
            &f.subjects,
            true,
            mixture,
        )
    };
    assert!(run(&model, &f.etas, false).is_some(), "control: checked");

    assert!(run(&model, &f.etas, true).is_none(), "a mixture fit");
    assert!(
        run(&model, &f.etas[..f.etas.len() - 1], false).is_none(),
        "EBEs that do not line up with the population"
    );
    assert!(
        explicit_h_check(
            &model,
            &population(),
            &f.params,
            &f.etas,
            &f.subjects[..f.subjects.len() - 1],
            true,
            false,
        )
        .is_none(),
        "per-subject results that do not line up with the population"
    );

    let mut fd = fast_pool_model("rk45");
    fd.gradient_method = GradientMethod::Fd;
    assert!(
        run(&fd, &f.etas, false).is_none(),
        "finite-difference gradients"
    );

    // Outside the recorder's scope the fit's own `H` was the finite-difference one, so the
    // comparison would be of a thing with itself: no `rhs_program`, no dual walk.
    let mut out_of_scope = fast_pool_model("rk45");
    out_of_scope.ode_spec.as_mut().unwrap().rhs_program = None;
    assert!(
        !crate::sens::ode_provider::adjoint::recorder_in_scope(&out_of_scope),
        "the fixture really is outside the recorder's scope"
    );
    assert!(
        run(&out_of_scope, &f.etas, false).is_none(),
        "a model the recorder does not serve"
    );

    let closed_form = parse_model_string(
        r#"
[parameters]
  theta TVCL(1.0, 0.1, 50.0)
  theta TVV(10.0, 1.0, 500.0)
  omega ETA_CL ~ 0.04
  sigma PROP ~ 0.04
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV
[structural_model]
  pk one_cpt_iv(cl=CL, v=V)
[error_model]
  DV ~ proportional(PROP)
"#,
    )
    .unwrap();
    assert!(
        run(&closed_form, &f.etas, false).is_none(),
        "a closed-form model"
    );
}

/// A pinned stiff method never takes an explicit step, so the gate declines before building any
/// finite-difference Jacobian.
#[test]
fn a_pinned_stiff_method_is_not_checked() {
    let model = fast_pool_model("rodas5p");
    let r = fit_with(&model);
    assert!(check_of(&model, &r, false).is_none());
    assert!(explicit_h_warning(&r).is_none());
}

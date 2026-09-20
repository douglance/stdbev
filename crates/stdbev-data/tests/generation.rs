//! Dataset generation and splitting tests.

#![allow(
    clippy::indexing_slicing,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::cast_precision_loss
)]

use std::collections::BTreeSet;

use stdbev_data::{Split, TrainingExample, check_all_populated, is_test_template, split_for};
use stdbev_types::OptionFormat;

fn generate(count: usize) -> (Vec<TrainingExample>, tempdir::Dir) {
    let dir = tempdir::Dir::new();
    stdbev_data::generate::generate(count, 7, dir.path(), OptionFormat::Compact)
        .expect("generates");
    let mut all = Vec::new();
    for split in Split::all() {
        let path = dir.path().join(format!("{}.jsonl", split.stem()));
        all.extend(stdbev_data::jsonl::read(&path).expect("reads back"));
    }
    (all, dir)
}

/// Minimal scratch directory helper, so the crate needs no dev-dependency for it.
mod tempdir {
    use std::path::{Path, PathBuf};

    pub struct Dir(PathBuf);

    impl Dir {
        pub fn new() -> Self {
            let base = std::env::temp_dir().join(format!(
                "stdbev-data-test-{}-{:?}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            std::fs::create_dir_all(&base).expect("temp dir");
            Self(base)
        }

        pub fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

#[test]
fn every_generated_example_is_valid() {
    let (all, _dir) = generate(900);
    assert_eq!(all.len(), 900);
    for example in &all {
        example
            .validate()
            .unwrap_or_else(|e| panic!("{}: {e}", example.id));
    }
}

#[test]
fn generation_is_deterministic_for_a_seed() {
    let (a, _d1) = generate(300);
    let (b, _d2) = generate(300);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.context, y.context, "same seed produced different data");
        assert_eq!(x.options, y.options);
        assert_eq!(x.target_probabilities, y.target_probabilities);
    }
}

#[test]
fn no_test_template_ever_appears_in_training_data() {
    // This is the whole point of the family holdout. If it leaks, the headline
    // accuracy measures memorization and predicts nothing.
    let dir = tempdir::Dir::new();
    stdbev_data::generate::generate(3000, 7, dir.path(), OptionFormat::Compact).expect("generates");
    let train = stdbev_data::jsonl::read(&dir.path().join("train.jsonl")).unwrap();
    let validation = stdbev_data::jsonl::read(&dir.path().join("validation.jsonl")).unwrap();
    let test = stdbev_data::jsonl::read(&dir.path().join("test.jsonl")).unwrap();

    let test_templates: BTreeSet<&str> = test.iter().map(|e| e.template.as_str()).collect();
    for example in train.iter().chain(&validation) {
        assert!(
            !test_templates.contains(example.template.as_str()),
            "template {:?} is in both training and test data",
            example.template
        );
    }
    assert!(!test_templates.is_empty(), "test split has no templates");
}

#[test]
fn all_three_splits_receive_examples() {
    let dir = tempdir::Dir::new();
    let report = stdbev_data::generate::generate(3000, 7, dir.path(), OptionFormat::Compact)
        .expect("generates");
    assert!(report.counts.train > 0);
    assert!(
        report.counts.validation > 0,
        "empty validation disables early stopping"
    );
    assert!(report.counts.test > 0);
}

#[test]
fn every_split_contains_all_three_primitives() {
    // A test split with no Noul examples would silently stop measuring a third of the
    // public API.
    let dir = tempdir::Dir::new();
    stdbev_data::generate::generate(6000, 7, dir.path(), OptionFormat::Compact).expect("generates");
    for split in Split::all() {
        let examples =
            stdbev_data::jsonl::read(&dir.path().join(format!("{}.jsonl", split.stem()))).unwrap();
        for prefix in ["choice", "noul", "score"] {
            assert!(
                examples.iter().any(|e| e.id.starts_with(prefix)),
                "{} split has no {prefix} examples",
                split.stem()
            );
        }
    }
}

#[test]
fn validation_shares_templates_with_training_by_design() {
    // Validation drives early stopping and temperature calibration, both of which need
    // in-distribution data. Test is where generalization is measured.
    let dir = tempdir::Dir::new();
    stdbev_data::generate::generate(3000, 7, dir.path(), OptionFormat::Compact).expect("generates");
    let train: BTreeSet<String> = stdbev_data::jsonl::read(&dir.path().join("train.jsonl"))
        .unwrap()
        .into_iter()
        .map(|e| e.template)
        .collect();
    let validation: BTreeSet<String> =
        stdbev_data::jsonl::read(&dir.path().join("validation.jsonl"))
            .unwrap()
            .into_iter()
            .map(|e| e.template)
            .collect();
    assert!(
        validation.iter().all(|t| train.contains(t)),
        "validation must stay in-distribution"
    );
}

#[test]
fn split_assignment_is_stable() {
    for template in ["ticket-plain", "noul-owner", "score-fields"] {
        let first = split_for(template, "group-a");
        for _ in 0..5 {
            assert_eq!(split_for(template, "group-a"), first);
        }
        // A test template stays test regardless of group.
        if is_test_template(template) {
            assert_eq!(split_for(template, "anything"), Split::Test);
        }
    }
}

#[test]
fn an_empty_split_is_reported_not_ignored() {
    assert!(check_all_populated(10, 0, 5).is_err());
    assert!(check_all_populated(10, 2, 5).is_ok());
}

#[test]
fn invalid_examples_are_rejected() {
    let base = TrainingExample {
        id: "x".into(),
        group: "g".into(),
        template: "t".into(),
        context: "c".into(),
        options: vec!["a".into(), "b".into()],
        target_probabilities: vec![0.5, 0.5],
    };
    assert!(base.validate().is_ok());

    let mut one_option = base.clone();
    one_option.options = vec!["a".into()];
    one_option.target_probabilities = vec![1.0];
    assert!(one_option.validate().is_err(), "1 option must be rejected");

    let mut mismatched = base.clone();
    mismatched.target_probabilities = vec![1.0];
    assert!(mismatched.validate().is_err());

    let mut unnormalized = base.clone();
    unnormalized.target_probabilities = vec![0.5, 0.9];
    assert!(unnormalized.validate().is_err());

    let mut negative = base.clone();
    negative.target_probabilities = vec![-0.1, 1.1];
    assert!(negative.validate().is_err());
}

#[test]
fn the_generator_emits_the_requested_option_format() {
    // Training data and serving must agree on the encoding. A corpus rendered verbose
    // and served compact silently asks the model a different question than it learned.
    let dir = tempdir::Dir::new();
    stdbev_data::generate::generate(600, 7, dir.path(), OptionFormat::Verbose).unwrap();
    let verbose = stdbev_data::jsonl::read(&dir.path().join("train.jsonl")).unwrap();
    assert!(
        verbose
            .iter()
            .filter(|e| e.id.starts_with("noul"))
            .all(|e| e.options[0].starts_with("FALSE")),
        "verbose Noul options should keep their descriptions"
    );

    let dir2 = tempdir::Dir::new();
    stdbev_data::generate::generate(600, 7, dir2.path(), OptionFormat::Compact).unwrap();
    let compact = stdbev_data::jsonl::read(&dir2.path().join("train.jsonl")).unwrap();
    assert!(
        compact
            .iter()
            .filter(|e| e.id.starts_with("noul"))
            .all(|e| e.options == vec!["no".to_string(), "yes".to_string()]),
        "compact Noul options should be bare polarity markers"
    );
}

#[test]
fn compact_generation_drops_descriptions() {
    let dir = tempdir::Dir::new();
    stdbev_data::generate::generate(600, 7, dir.path(), OptionFormat::Compact).unwrap();
    let rows = stdbev_data::jsonl::read(&dir.path().join("train.jsonl")).unwrap();
    for e in rows.iter().filter(|e| e.id.starts_with("choice")) {
        for o in &e.options {
            assert!(
                !o.contains(':') && !o.contains('\n'),
                "compact Choice options must be bare labels, got {o:?}"
            );
        }
    }
}

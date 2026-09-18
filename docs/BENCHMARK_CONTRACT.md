# Benchmark contract

A benchmark result is meaningful only when the prediction target, biological grouping unit, and acquisition domain are explicit.

## Required fields

- **features:** frozen embedding dimensions only
- **label:** binary downstream target
- **group:** donor, patient, or other biological unit that must not cross folds
- **domain:** optional site, scanner, batch, cohort, or acquisition source

## Group-held-out score

The primary score is computed using stratified group cross-validation. Repeated samples from one donor remain on one side of a fold.

## Naive score and leakage gap

A conventional stratified row-wise split is also evaluated. The difference

`naive ROC AUC - grouped ROC AUC`

is reported as the leakage gap. It is an audit statistic, not proof of leakage by itself.

## Domain shift

When domains are supplied, the benchmark trains on all but one domain and evaluates on the held-out domain. This approximates deployment under acquisition shift.

## Probe restriction

The benchmark uses a standardized linear logistic probe. Keeping the downstream model deliberately simple helps isolate the accessibility and transferability of information already present in the embedding.

## Calibration

Brier score is reported alongside discrimination metrics because a representation can rank examples well while still producing poorly calibrated probabilities.

## Interpretation limits

High probe performance does not establish clinical utility. External validation, prospective evaluation, prevalence-aware calibration, and decision-analytic endpoints remain separate requirements.

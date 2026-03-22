# MLOps Playbook

## What Is Monitored in Deployment
The API runtime monitor tracks:
- request volume,
- request error rate,
- p50/p95/max latency,
- throughput in requests per minute,
- per-endpoint latency and error rate,
- CPU and memory usage,
- GPU memory usage when CUDA is available,
- estimated infrastructure cost per 1000 requests.

This means the serving layer is monitored across four operational planes:
- service health: request count, endpoint behavior and error rate,
- performance: latency percentiles and throughput,
- infrastructure: CPU, memory and GPU usage,
- economics: estimated cost per 1000 requests.

The active operating policy lives in `src/mlops/policies/default_operating_policy.yaml`.

## Retraining Triggers
Retraining is not driven only by data drift. The project now distinguishes four trigger families:

1. Periodic retraining
Rebuild the challenger on a fixed cadence even if nothing looks broken. This protects against stale models and hidden data change.

2. KPI-driven retraining
Trigger retraining when a production KPI such as Dice, mIoU, manual acceptance rate, or downstream clinical utility falls outside the approved band versus the current champion.

3. Drift-driven retraining
Trigger retraining when the serving distribution diverges from the approved baseline according to PSI, KS or related shift metrics.

4. Rollback
Rollback is not retraining. It is an operational safety action used when error rate, latency or repeated incident windows indicate that the current model should be replaced immediately by the previous champion.

## MLflow Tracking Standard
Each training run should log:
- hyperparameters,
- epoch metrics,
- dataset manifest,
- checkpoint artifact,
- model card,
- packaging manifest,
- serving files and Docker packaging files,
- data contracts and dataset source metadata.

This makes the training run auditable from dataset version and quality state to the exact serving package that was deployed.

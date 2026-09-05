# Scientific Artifact Policy

## Purpose

Experiment outputs must remain traceable without allowing large generated
binaries to accumulate silently in normal Git history.

## Normal Git

The following are expected to be committed when scientifically relevant:

- experiment configurations;
- run.yaml provenance records;
- training and evaluation logs;
- epoch-level metrics;
- prediction tables;
- artifact manifests;
- SHA-256 checksum files;
- small diagnostic summaries;
- selected figures used in analysis or reporting.

## Local canonical artifacts

The following remain outside normal Git by default:

- model checkpoints;
- canonical Grad-CAM .npy maps;
- bulk rendered Grad-CAM images;
- intermediate activation tensors;
- large feature dumps;
- other generated model binaries.

Each canonical binary artifact must still be represented in a tracked
artifact manifest containing enough information to identify it, including
its path, SHA-256, run ID and relevant source/checkpoint identity.

## Cross-machine handling

A machine writes artifacts locally first.

Stable artifacts may then be synchronized explicitly between machines.
The artifact manifest and SHA-256 are used to verify identity after transfer.

No experiment should rely on an unverified copied artifact.

## Git-LFS / DVC

Git-LFS or DVC is not required at this stage.

Introduce one only if repeated cross-machine or archival handling of large
canonical artifacts makes the additional dependency worthwhile.

## Report figures

Bulk run-local visualizations remain local.

A figure selected for dissertation/report use should be copied into an
explicit tracked reporting location and retain provenance linking it to:

figure
-> visualization hash
-> numerical/localization artifact
-> source image hash
-> checkpoint hash
-> run ID
-> Git commit
-> frozen FantasyID split.

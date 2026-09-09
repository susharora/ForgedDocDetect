# Tech-2 Decision Log — ResNet-18 + Grad-CAM

This document records scientific design decisions for the FantasyID
ResNet-18 + Grad-CAM pipeline.

The decision log explains *why* the frozen scientific configuration has
its present values. The executable contract remains:

`configs/experiments/resnet18_gradcam.yaml`

Date of current protocol freeze: 2026-09-09.

---

## D1 — Preserve the frozen Tech-1 project split

**Status:** FROZEN

**Decision**

Use the frozen Tech-1 split without alteration:

- project_train: 160 cards / 1,440 images
- dev_val: 51 cards / 459 images
- project_train ↔ dev_val card overlap: 0
- held-out test remains inaccessible during model development

`dev_val` is the sole model-selection partition.

**Reason**

The split was established before Tech-2 model development and is
card-disjoint between project_train and dev_val. Repartitioning after
observing model behaviour would introduce an avoidable model-selection
degree of freedom.

---

## D2 — Retain `stem_family` terminology and validate dev composition

**Status:** FROZEN

**Decision**

Use the Tech-1 derived term `stem_family`; do not silently relabel it as
a formal document-template identifier.

All ten observed stem families must occur in both project_train and
dev_val.

The frozen dev allocation must continue to match the proportional
largest-remainder allocation used by Tech-1.

**Evidence**

`logs/audit_split_stem_family_2026-09-09_164053_260036Z.csv`

SHA-256:

`1b9b77a00c1fd3588430b81cdefe446a944886946077fc2623a334019b162ed8`

Audit result:

- 10 / 10 families represented in project_train
- 10 / 10 families represented in dev_val
- all reconstructed dev quota deltas = 0
- source cards reconcile to 211
- project_train cards reconcile to 160
- dev_val cards reconcile to 51

---

## D3 — Primary ResNet remains an image-level classifier

**Status:** FROZEN

**Decision**

The primary ResNet-18 receives only image-level binary labels during
training.

The primary pipeline does not use:

- localisation supervision
- segmentation supervision
- targeted region dropout
- patch labels
- Grad-CAM supervision

**Reason**

The intended comparison distinguishes a conventional classifier with
post-hoc Grad-CAM attribution from a forensic model with native
localisation capability.

Injecting localisation supervision into the primary ResNet would weaken
that methodological distinction.

Targeted region dropout remains a possible annotation-informed
ablation, not part of the primary classifier.

---

## D4 — Binary-class polarity

**Status:** FROZEN

**Decision**

Use:

- bonafide = class 0
- attack = class 1
- positive class = attack

**Reason**

This preserves the existing FantasyID/Tech-1 project convention and
prevents later metric or threshold polarity ambiguity.

---

## D5 — Candidate preprocessing geometry

**Status:** FROZEN FOR DEVELOPMENT SCREENING

**Decision**

Screen two input resolutions using the same geometry policy:

### r256

- content height: 256 px
- canvas: 256 × 432

### r512

- content height: 512 px
- canvas: 512 × 864

For both:

1. decode with Pillow;
2. convert explicitly to RGB;
3. no EXIF orientation correction, based on the completed decode audit;
4. resize while preserving aspect ratio;
5. no centre crop;
6. no square distortion;
7. bilinear interpolation;
8. antialiasing enabled;
9. horizontally centre-pad to the fixed canvas;
10. fail if resized content unexpectedly exceeds the canvas width;
11. use ImageNet normalization.

Padding is performed in pre-normalized RGB using the ImageNet channel
mean:

- R = 0.485
- G = 0.456
- B = 0.406

which maps to normalized zero.

**Reason**

The observed documents are landscape with a narrow aspect-ratio range.
Standard ImageNet square centre-cropping would remove a substantial
fraction of document width, while direct square resizing would distort
document geometry.

The fixed landscape canvases preserve the whole document and permit
deterministic batching.

---

## D6 — Baseline is resampling-constant, not resampling-free

**Status:** FROZEN

**Decision**

The primary baseline uses deterministic resize/pad preprocessing and no
stochastic image augmentation.

The following are disabled in the initial baseline:

- horizontal flip
- random crop
- random resized crop
- random rotation
- affine jitter
- colour jitter
- random erasing
- JPEG recompression
- blur
- added noise
- MixUp
- CutMix

**Reason**

The deterministic resize is itself a global resampling operation.
Therefore the scientifically correct description is:

> the baseline is resampling-constant: every image in both classes
> receives the same deterministic preprocessing transform, with no
> stochastic resampling augmentation.

Many standard augmentations can create, suppress, or alter the
low-level resampling and boundary evidence relevant to document
forensics. Their effect should therefore be tested explicitly as
ablations rather than assumed beneficial.

---

## D7 — Use FP32 batch size 32 at both candidate resolutions

**Status:** FROZEN

**Decision**

Use:

- batch size = 32
- dtype = FP32
- AMP = disabled

for both 256×432 and 512×864.

Machine-specific DataLoader worker count remains outside the scientific
configuration.

**Reason**

Keeping batch size fixed prevents resolution and optimization-noise
changes from being confounded.

The worst-case proposed Stage-B configuration was empirically tested:

- full ResNet-18 backbone trainable
- BatchNorm in train mode
- batch = 32
- input = 3 × 512 × 864
- FP32
- AdamW
- forward + backward + optimizer step

and passed on both machines.

**Evidence — home**

Machine:

`sush-AMD / NVIDIA GeForce RTX 5090`

Artifact:

`logs/probe_resnet18_training_memory_sush-AMD_2026-09-09_170616_974489Z.yaml`

SHA-256:

`b3714653d121e6b195753f870e84286946251a9257280845b657566c32929432`

Peak reserved VRAM:

approximately 28.36% of physical VRAM.

**Evidence — lab**

Machine:

`IMTA134 / NVIDIA RTX PRO 5000 Blackwell`

Artifact:

`logs/probe_resnet18_training_memory_IMTA134_2026-09-09_170948_841635Z.yaml`

SHA-256:

`68a752aad899a623a86ff0c5aa0c7dd89787d936057fd43292213037cf420b70`

Peak reserved VRAM:

approximately 18.60% of physical VRAM.

Both satisfy the predeclared 80% maximum-reserved-memory guard.

---

## D8 — Use ImageNet-pretrained ResNet-18 and fine-tune the full backbone

**Status:** FROZEN

**Decision**

Initialize with:

`torchvision ResNet18_Weights.IMAGENET1K_V1`

Checkpoint SHA-256:

`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`

Replace the original ImageNet classifier with a new 512 → 2 linear
layer.

Do not permanently freeze the pretrained backbone.

**Reason**

The project_train set contains only 1,440 images, making transfer
learning preferable as the primary recipe.

The completed pretrained representation audit also showed that
document-edit differences already propagate through the pretrained
network. The relevant development question is therefore how best to
adapt the representation, rather than assuming that the backbone must
be learned from scratch.

A from-scratch ResNet-18 remains a possible later ablation.

---

## D9 — Use class-weighted cross entropy

**Status:** FROZEN

**Decision**

Derive weights from project_train only:

`w_c = N / (2 * n_c)`

With:

- N = 1,440
- bonafide = 480
- attack = 960

the frozen weights are:

- bonafide: 1.50
- attack: 0.75

Use the same project_train-derived weights for:

- training loss
- Stage-A dev loss
- Stage-B dev loss
- checkpoint selection
- LR selection
- resolution comparison

Do not recompute weights from dev_val.

Do not combine class weighting with a weighted sampler.

**Reason**

This corrects the 2:1 class imbalance while preserving the actual
training sample distribution.

Using the same weighted loss for model fitting and model selection makes
the optimization and selection objectives explicit and consistent.

---

## D10 — Weighted epoch loss must be computed over the complete dataset

**Status:** FROZEN

**Decision**

Define epoch weighted CE as:

`sum(sample_weight * per_sample_CE) / sum(sample_weight)`

over the complete epoch.

Do not estimate epoch loss by taking an unweighted arithmetic mean of
batch-level weighted losses.

**Reason**

The last batch can contain fewer examples. Dataset-level numerator and
denominator accumulation gives the exact configured objective
independently of batch partitioning.

---

## D11 — Stage A uses validation-plateau head adaptation

**Status:** FROZEN

**Decision**

Stage A:

- backbone parameters: `requires_grad = False`
- backbone module mode: `eval`
- backbone BatchNorm running statistics: frozen
- new FC head: trainable
- optimizer contains FC parameters only
- FC LR = 1e-3
- maximum epochs = 10
- plateau patience = 2
- meaningful relative improvement = 0.5%
- stopping metric = class-weighted dev CE
- checkpoint = raw minimum class-weighted dev CE

For each resolution+seed, execute Stage A once.

All Stage-B LR branches for that resolution+seed must begin from the
exact same best Stage-A checkpoint.

**Reason**

A fixed one-epoch warm-up would be arbitrary.

The purpose of Stage A is to allow the new randomly initialized binary
head to adapt to the existing representation before the pretrained
backbone is changed.

Both `requires_grad=False` and evaluation mode are required: freezing
parameters alone would still permit BatchNorm running statistics to
change.

---

## D12 — Stage B performs full domain adaptation

**Status:** FROZEN

**Decision**

Stage B starts from the exact raw-argmin Stage-A checkpoint.

Then:

- create a fresh optimizer
- unfreeze the complete backbone
- put backbone in train mode
- train BatchNorm affine parameters
- permit BatchNorm running statistics to adapt to FantasyID
- keep FC trainable
- FC LR = 1e-3
- maximum epochs = 30
- plateau patience = 5
- meaningful relative improvement = 0.5%
- stopping metric = class-weighted dev CE
- checkpoint = raw minimum class-weighted dev CE

Stage-B epoch 1 must record that BatchNorm has switched from frozen to
domain-adapting state.

**Reason**

The primary recipe is intended to adapt the complete ImageNet feature
hierarchy to FantasyID rather than using ResNet-18 only as a fixed
feature extractor.

The BatchNorm transition can transiently change validation behaviour;
patience 5 deliberately permits recovery from that transition.

---

## D13 — AdamW optimizer contract

**Status:** FROZEN

**Decision**

Use AdamW with:

- beta1 = 0.9
- beta2 = 0.999
- eps = 1e-8
- weight decay = 1e-4
- AMSGrad = false
- foreach = false
- fused = false
- LR schedule = constant

Exclude from weight decay:

- all bias parameters
- all BatchNorm affine parameters

**Reason**

Explicitly encoding normally hidden optimizer defaults removes
framework-version ambiguity.

A constant schedule keeps the initial learning-rate comparison
interpretable and avoids adding warm-up/minimum-LR/schedule-duration
hyperparameters before evidence shows they are needed.

---

## D14 — Early-stopping tolerance and checkpoint selection are separate

**Status:** FROZEN

**Decision**

The 0.5% relative-improvement criterion controls only the early-stopping
patience counter.

It does not control checkpoint saving.

For every epoch:

- if weighted dev loss is the raw lowest value seen so far, save it as
  the best checkpoint;
- reset the patience anchor only after at least 0.5% relative
  improvement from the previous patience anchor.

Therefore:

- stopping checkpoint = raw argmin weighted dev loss
- patience criterion = 0.5% relative improvement

**Reason**

Small genuine loss improvements should not be discarded merely because
they are too small to justify extending training indefinitely.

---

## D15 — AUROC is a diagnostic guard, not a second selection criterion

**Status:** FROZEN

**Decision**

Record dev AUROC every epoch.

The selected checkpoint remains the raw weighted-dev-loss argmin.

For each Stage-B run compute:

`best dev AUROC in run - AUROC at loss-argmin checkpoint`

If this difference is strictly greater than 0.01:

- set a protocol-review flag;
- do not automatically switch checkpoints.

**Reason**

On the 459-image dev set, weighted CE and AUROC can meaningfully
disagree.

The discrepancy should be surfaced rather than silently resolved by an
undeclared secondary selection rule.

---

## D16 — Predeclare a small backbone-LR screen

**Status:** FROZEN

**Decision**

At screening seed 8, test for each resolution:

- backbone LR = 3e-5
- backbone LR = 1e-4
- backbone LR = 3e-4

with fixed:

- FC LR = 1e-3
- all other training settings unchanged

For each resolution:

1. run Stage A once;
2. fork all three Stage-B candidates from the same Stage-A checkpoint;
3. select the LR whose raw best checkpoint has the lowest weighted dev
   CE;
4. exact numerical ties select the lower backbone LR.

**Reason**

This converts the backbone LR from a merely conventional choice into a
small predeclared development experiment while avoiding a large
hyperparameter search.

---

## D17 — Carry the best LR per resolution into paired multi-seed analysis

**Status:** FROZEN

**Decision**

For each resolution separately, carry its selected backbone LR into:

- seed 8
- seed 9
- seed 10

Seed 8 screening results may be reused.

Each run seed controls:

- Python RNG
- NumPy RNG
- torch CPU RNG
- torch CUDA RNG
- classifier initialization
- DataLoader shuffle generator
- DataLoader worker seeding

Use deterministic algorithms with:

- cuDNN benchmark = false
- cuDNN deterministic = true
- TF32 disabled
- deterministic-algorithm enforcement enabled
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`

**Terminology**

This is a multi-seed stability analysis, not an independent hold-out
confirmation experiment.

---

## D18 — Resolution selection is an asymmetric practical non-inferiority rule

**Status:** FROZEN

**Decision**

512×864 is the a-priori preferred resolution because it retains more
source information and provides a denser final convolutional spatial
grid for Grad-CAM.

Classification performance therefore acts as a practical
non-inferiority gate rather than a symmetric winner-takes-all race.

For each paired seed `s`:

`d_s = L_512(s) - L_256(s)`

where `L` is that resolution's raw minimum class-weighted dev CE using
its selected LR.

Then:

`mean_d = mean(d_8, d_9, d_10)`

and:

`margin = 0.05 * mean(L_256(8), L_256(9), L_256(10))`

Decision:

- select 512×864 if `mean_d <= margin`
- select 256×432 if `mean_d > margin`

Report the sign of each individual `d_s`, but the sign pattern does not
change the decision rule.

**Reason**

The case for higher resolution was established before observing
fine-tuning results: small forensic traces can be destroyed by
downsampling, and the final spatial activation grid is substantially
denser at 512.

A tiny random loss difference should therefore not silently remove the
resolution preferred for the localisation experiment.

This is a predeclared practical decision rule, not a formal statistical
non-inferiority hypothesis test.

---

## D19 — Test access occurs only after development configuration is frozen

**Status:** FROZEN

**Decision**

The held-out test remains unavailable for:

- preprocessing selection
- LR screening
- multi-seed stability analysis
- resolution selection
- checkpoint selection
- augmentation decisions
- Grad-CAM target-layer selection
- threshold design

After the primary configuration is frozen, all three selected-resolution
seed checkpoints proceed to final detection evaluation.

Report:

- each seed individually
- mean
- min
- max

Do not select the best test seed.

---

## D20 — Threshold is derived separately for each selected seed checkpoint

**Status:** FROZEN

**Decision**

For each of the three final seed checkpoints:

1. derive its operating threshold from its own normal dev_val scores;
2. use the frozen Tech-1 metric and threshold convention;
3. target FPR = 10%;
4. apply that threshold unchanged during final test evaluation.

Threshold selection does not participate in checkpoint selection.

**Reason**

Class weighting can change score calibration. A fixed probability such
as 0.5 is therefore not assumed to be the desired operating point.

---

## D21 — Representative Grad-CAM checkpoint is the median seed

**Status:** FROZEN

**Decision**

For the selected resolution, choose one representative checkpoint for
the principal Grad-CAM analysis:

the seed whose raw best weighted dev loss is the median of the three
seed losses.

If two values are exactly tied, choose the lower seed number.

**Reason**

The principal localisation figure should not use whichever seed happened
to obtain the luckiest development result.

All three checkpoints remain available for detection evaluation.

---

## D22 — Do not freeze the Grad-CAM target layer before classifier validation

**Status:** FROZEN AS A DEFERRED DECISION

**Decision**

Grad-CAM is not used to train or select the classifier.

The canonical Grad-CAM target layer remains unfrozen until the selected
classifier is validated.

Later raw localization maps will be stored separately from rendered
visualizations, with the intended canonical raw-map contract:

- float32
- H × W
- normalized to [0, 1]

**Reason**

Selecting the Grad-CAM layer before understanding the fine-tuned
classifier would prematurely mix localization decisions into classifier
development.

---

## D23 — Defer region-occlusion geometry and targeted region dropout

**Status:** DEFERRED OPTIONAL ABLATION

**Decision**

Do not define or freeze `delta_face` / `delta_text` values during the
primary classifier-development stage.

If semantic-region occlusion is later activated:

- mask geometry must be derived from project_train only;
- dilation must be expressed relative to source image height;
- mask geometry is used only for occlusion/ablation construction;
- localization ground-truth geometry is never dilated;
- candidate margins must undergo coverage and collateral-occlusion
  validation before freezing.

The same deferral applies to masked-dev counterfactual diagnostics.

**Reason**

These quantities are required only if the optional region-dropout or
occlusion diagnostic branch is activated. They should not block or
contaminate the annotation-free primary classifier.

---

## D24 — Other secondary experiments remain explicit ablations

**Status:** DEFERRED

The following are not part of primary model selection:

- JPEG / blur / noise augmentation
- high-pass or SRM-style input
- patch-level training
- from-scratch ResNet-18
- identity-adversarial training
- auxiliary contrastive objectives

They may be activated later only as explicitly named experiments with
their own predeclared protocol.

---

# Protocol-freeze evidence

Before this decision log was frozen, the scientific schema-v2 contract
was validated on both development machines.

The validator checks:

- frozen Tech-1 provenance
- scientific class contract
- project_train/dev class counts
- project_train-derived class weights
- preprocessing policy
- batch size and memory-probe evidence
- Stage-A and Stage-B semantics
- optimizer details
- stopping/checkpoint separation
- LR-screening plan
- multi-seed plan
- resolution-selection rule
- threshold policy
- deferred-work boundaries
- held-out-test protection

The validator itself writes timestamped canonical evidence to `./logs/`.

After changing the experiment protocol status from
`ready_for_validation` to `frozen`, the validator must be run again from
a clean Git commit so that final validation evidence refers to the exact
frozen scientific YAML rather than the pre-freeze YAML.

---

# Freeze boundary

The decisions above define the primary ResNet-18 transfer-learning
development protocol.

Changes to any FROZEN item after this boundary require:

1. an explicit new decision-log entry explaining the reason;
2. a scientific-config change;
3. a new config SHA-256;
4. re-validation;
5. no retroactive use of held-out test evidence to justify the change.

The optional/deferred experiments do not become active merely by being
listed here.

---

## D25 — Restart Stage-B branch RNG state from the run seed

**Status:** FROZEN

**Decision**

For every Stage-B branch for a given resolution + run seed + backbone
learning rate:

1. reapply the frozen run-level reproducibility configuration using the
   same `run_seed`;
2. construct fresh project_train and dev_val DataLoaders whose generators
   are seeded from that `run_seed`;
3. construct a fresh ResNet-18 instance;
4. restore the exact raw-argmin Stage-A model parameters and buffers;
5. apply the Stage-B full-backbone training contract;
6. construct a fresh Stage-B AdamW optimizer.

The Stage-A -> Stage-B boundary carries only the selected model state.

Do not carry:

- Stage-A AdamW state;
- Python RNG state;
- NumPy RNG state;
- torch CPU/CUDA RNG state;
- project_train DataLoader-generator state;
- dev_val DataLoader-generator state.

All Stage-B LR candidates for a given resolution and seed must therefore
begin with:

- the same Stage-A raw-best model state;
- the same run-level RNG initialization;
- the same project_train ordering for corresponding Stage-B epochs.

**Reason**

The backbone learning rate is the variable being compared during the
Stage-B screening experiment.

Continuing mutable Stage-A RNG/DataLoader state would make the Stage-B
batch sequence depend on how many Stage-A epochs happened to execute and
on which Stage-A epoch became the raw-best checkpoint.

Resetting every Stage-B branch from the frozen `run_seed` isolates the LR
comparison from Stage-A stopping duration and gives all LR candidates the
same minibatch-order exposure.

This is also consistent with Stage B already being defined as a new
optimization stage with a fresh AdamW optimizer.

The selected Stage-A model state remains the only scientific state
carried across the stage boundary.
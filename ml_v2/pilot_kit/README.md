# TrueCost V2 development pilot kit

This kit prepares 32 reviewed, previously unused task groups (four per family), with the original two Qwen model checkpoints and seeds 440/441: 128 DEVELOPMENT attempts. These groups are excluded from future calibration and final evaluation. This integration pilot does not establish V2 predictive accuracy.

V1 source files, frozen manifests, collected traces, and models are read-only inputs. Installation creates a new sibling folder: `~/truecost_roar/TrueCost_V2_Development_Pilot`. It refuses existing destinations and rechecks the original selection and source hashes before creating the pilot. The original worker, tool harness, transport, server, and shard-audit code are copied without modification. The new manifest/orchestration code supports a 32-group pilot rather than V1's hardcoded 3,000-group campaign.

## What is selected

The original candidate/review/exclusion snapshot yields 162 records in 159 groups. The pilot selects four unique groups per family by deterministic matching, with natural input-band preferences and a fixed SHA-256 task tie-break. The matching step prevents families that share groups from silently consuming one another's quota. The selection has 28 tasks at <=512 source-bearing tokens, one at 513–2048, two at 2049–4096, and one at 4097–8192. Full task IDs, family/source metadata and selection checks are in `payload/tasks.json` and `payload/selection_receipt.json`.

All four currently eligible ideation groups are consumed as development. Only four of eight eligible tabular groups are selected. The remaining reviewed pool is insufficient for large independent family-specific calibration claims. Existing relation reviews and exclusions reduce known overlap; they do not rule out every semantic overlap or model pretraining exposure.

## Install on Roar

Download this ZIP to your Mac, then upload it in your Mac terminal:

```bash
scp ~/Downloads/TrueCost_V2_Pilot_Kit.zip kvr5648@submit01.2e.hpc.psu.edu:~/
```

In the Roar terminal:

```bash
source ~/truecost_roar/truecost_analysis_venv/bin/activate
python --version
unzip -q ~/TrueCost_V2_Pilot_Kit.zip -d ~/truecost_roar
cd ~/truecost_roar/TrueCost_V2_Pilot_Kit
python install_pilot.py
cd ~/truecost_roar/TrueCost_V2_Development_Pilot
python pilot.py preflight
```

Python >=3.9 is required. Installation also writes `provenance/development_exclusions.json` with pilot task/group/source-group IDs and normalized text/context fingerprints. Future V2 selection must apply that registry through its grouping graph. V1 exclusion files are not modified.

The original verified tokenizer bytes are copied from the source project into the pilot and rehashed; only their registry paths are relocated. The Python environment must contain `tokenizers`, and `apptainer` must be available on PATH. If the analysis environment lacks either, activate/load the original collection environment rather than changing V1 code. The kit does not install dependencies or download models.

Preflight checks the frozen pilot, copies' hashes, model context admission and size bands, Ollama binary hash, and container-image hash. It first uses the existing `ROAR_OLLAMA_BIN`, `ROAR_TOOL_IMAGE`, and `OLLAMA_MODELS` variables (or saved pilot paths). If missing, it searches the existing Roar workspace and standard local Ollama directories for uniquely matching assets. If discovery is ambiguous or incomplete, it stops and names the missing variables. Do not guess paths or weaken the checks; return the exact error for diagnosis.

If preflight succeeds, submit from the pilot folder:

```bash
python pilot.py submit
python pilot.py status
```

Submission uses account `open`, partition `standard`, one V100 GPU, four CPUs, 32 GB RAM, and four hours per array task. There are four tasks of 32 assignments each, with at most two running concurrently. This caps a single full-array submission at 16 allocated GPU-hours if all four consume the full walltime; actual usage depends on execution and scheduling. No new allocation or job has been submitted by preparing this kit.

The batch script uses the exact submitting Python executable and clears inherited container binds. Inference refuses login nodes, verifies model digests and full GPU residency through the copied original collector, and uses the original bounded controller (six requests, 1,024 output tokens/request, 16,384 context, temperature zero). Original warmup calls are operational checks, excluded from research totals.

## Progress, pausing, and clean continuation

Logs are `v2-pilot-ARRAYJOB_ARRAYINDEX.log` in the new pilot folder. Status performs the original strict shard audit and prints planned/complete/pending counts. A shard with incomplete or conflicting evidence is reported as blocked. Preserve the error/log/evidence and return it for inspection; do not delete an attempt or retry it to obtain a preferred outcome.

To request a pause between attempts:

```bash
touch frozen/PAUSE_REQUESTED
```

The collector also stops before a new attempt if fewer than 2,100 seconds remain before the actual Slurm end time. Do not cancel an active attempt merely to pause; cancellation can leave incomplete telemetry. Each attempt has a 1,800-second deadline. The pilot may require continuation if its 32 assignments do not fit the allocated walltime.

After a clean stop and after the old array has ended, remove your pause marker if one was deliberately created, then run:

```bash
python pilot.py submit --resume
```

Resume refuses a still-active previous array, audits retained work, submits only pending shards, and skips already completed attempts. Any incomplete attempt or failure record blocks resume. There are no automatic inference retries and no repetition-error bypass. Stale lock files are not removed automatically.

## Export results

When status shows all 128 complete and audited:

```bash
python pilot.py export
```

This creates `TrueCost_V2_Pilot_Results.zip` in the new pilot folder, containing the audited CSV, manifest/tasks/config, audit counts, provenance, and raw attempt evidence with hashes. The export refuses partial/blocked work and will not overwrite an existing archive. The original raw source snapshots, Ollama binary, model weights and container image are not copied into the export.

Download in your Mac terminal:

```bash
scp kvr5648@submit01.2e.hpc.psu.edu:/storage/home/kvr5648/truecost_roar/TrueCost_V2_Development_Pilot/TrueCost_V2_Pilot_Results.zip ~/Downloads/
```

Upload that results ZIP to continue analysis. It remains development data; it must not be used as an untouched final test for V2.

## Verification and limits

Local offline tests use synthetic transports/counters and mock tool evidence. They do NOT run generated code, GPU inference, a real tokenizer, Slurm, or Apptainer. They exercise all 128 planned assignments across all families, original audit acceptance, raw-evidence tampering rejection, incomplete-attempt resume blocking, quota/group/review gates, reference exclusion from public tasks, locks, deadline admission, and a staged installer/no-overwrite/source-unchanged fixture. Source selection was recomputed against the actual uploaded metadata and execution-source copies were compared with the original frozen hashes.

The installation fixture mocks source selection and uses a fake tokenizer file solely to test installation mechanics. Actual installation repeats the full original-source checks and requires the original verified tokenizer registry/files. Tokenizer admission and runtime/container integration are necessarily deferred to Roar preflight and the development pilot. No claims of V2 prediction improvement are made by these software tests.

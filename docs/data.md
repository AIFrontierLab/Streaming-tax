# Getting the code, checkpoints and data

No third-party code, checkpoint or dataset ships with this repository (NOTICE lists the few short texts it quotes).
Three commands fetch them, each at the exact version behind the paper's numbers, and check every file:

```bash
streamreplace fetch-upstream all                 # third-party code (official scorers, streaming runtimes): ~300 files
streamreplace download models all                # the 11 paper models + the SigLIP vision tower (12 checkpoints)
streamreplace download judges all                # extraction judge, HarmBench classifiers, RJScore judge, Lingo-Judge
streamreplace download datasets paper-core       # the 15 main benchmarks; paper-supp and finetune for the others
```

Everything lands under one workspace, `$STREAMREPLACE_HOME` (default `workspace/` in the checkout). Each root can be
moved on its own (`STREAMREPLACE_UPSTREAM`, `STREAMREPLACE_CHECKPOINTS`, `STREAMREPLACE_DATA`, or a paths file with
`--paths FILE`); a paths file can also point single checkpoints or datasets at existing copies
(`checkpoint_dirs: {<org>/<name>: DIR}`, `dataset_dirs: {<benchmark>: DIR}`). `streamreplace doctor` shows the
resolved roots and what is present.

Check first, download later: `--dry-run` resolves every pin against its publisher (Hugging Face API at the pinned
revision, GitHub at the pinned commit, file sizes of fixed URLs, Google Drive ids) and prints where each file would
go, without downloading anything. `--dry-run --offline` prints the plan without any network request.
`--verify-only` checks what is already on disk.

## Third-party code: `streamreplace fetch-upstream`

`manifests/upstream.yaml` pins 17 GitHub repositories to full commits and lists exactly the files used, each with
its git blob SHA-1 (what `git hash-object FILE` prints). `fetch-upstream` downloads those files into
`<upstream root>/<name>/<path in the repository>` (default `$STREAMREPLACE_HOME/upstream`) and verifies each one;
a repository's tree replaces the old one only once all its files verify. Repositories:

| Group | Entries |
|---|---|
| core (P/C/S evaluation and scoring) | harmbench, vmdt, llava (LLaVA-NeXT), mmduet, mmduet2, thinkstream, livestar, tomato, timeblind, videomme (Video-MME's evaluation script), videoreasonbench, video_safetybench, mcv_safetybench |
| generalization (robustness, fine-tuning) | openuav_qa, lingoqa, promptbench, vlmevalkit |

```bash
streamreplace fetch-upstream --list              # what is pinned, with licences
streamreplace fetch-upstream core                # one group, or names: fetch-upstream llava mmduet
streamreplace fetch-upstream --verify-only       # offline check of the fetched files
streamreplace fetch-upstream all --from DIR      # offline machine: copy from checkouts DIR/<name>/ (still verified)
```

Three upstream files carry small edits the paper's runs used; the manifest stores them as text replacements with the
reason (`patches`), and the fetcher applies them only to the exact upstream blob they were written for:
LLaVA-NeXT's vision-tower builder dispatches local SigLIP folders before the generic CLIP branch; ThinkStream's
last window also delivers a question stamped exactly at its end (as its own comment says it should); the
VideoReasonBench judge parses move lists with `ast.literal_eval` instead of `eval` and fails a malformed list
instead of crashing. AURA and MOSS-VL are listed for provenance only and never fetched (the AURA adapter quotes its
system prompt; NOTICE lists every quoted text).

The default method downloads each file from `raw.githubusercontent.com`; `--method tarball` takes one archive per
repository instead (more data, fewer requests), and the default falls back to it when per-file downloads fail. Set
`GITHUB_TOKEN` if GitHub rate-limits your address.

## Checkpoints: `streamreplace download models|judges`

The checkpoint references live in `configs/models.yaml` and `configs/judges.yaml` as `<repo>@<revision>`;
`manifests/checkpoints.yaml` lists the same checkpoints with their licences and must agree (a revision that differs
is refused). A checkpoint goes to `<checkpoints root>/<org>/<name>` through `huggingface_hub.snapshot_download` at
exactly that revision, and `.revision` in the folder records it (`doctor` compares it with the configuration).

| Model id | Checkpoint (Hugging Face) | Notes |
|---|---|---|
| llava_ov_7b | lmms-lab/llava-onevision-qwen2-7b-ov @0b07bf75 | also MMDuet's base model |
| (vision tower) | google/siglip-so400m-patch14-384 @9fdffc58 | LLaVA-OneVision and MMDuet |
| mmduet_7b | wangyueqian/MMDuet @23669174 | LoRA + heads on LLaVA-OneVision-7B |
| qwen25vl_3b | Qwen/Qwen2.5-VL-3B-Instruct @66285546 | |
| thinkstream_3b | CASIA-IVA-Lab/ThinkStream-3B @9bc0b5aa | |
| mmduet2_3b | wangyueqian/MMDuet2 @3224235e | |
| qwen3vl_8b | Qwen/Qwen3-VL-8B-Instruct @0c351dd0 | |
| aura_8b | aurateam/AURA @88ee5506 | |
| livestar_iv_8b | yzy666/LiveStar_InternVideo_8B @902503c3 | weights only; code from `fetch-upstream livestar` |
| livestar_8b | yzy666/LiveStar_8B @fe12f966 | weights only; code from `fetch-upstream livestar` |
| moss_vl | OpenMOSS-Team/MOSS-VL-Instruct-0708 @abd21853 | |
| moss_vl_rt | OpenMOSS-Team/MOSS-VL-Realtime @06b06761 | |

| Judge id | Checkpoint | Used for |
|---|---|---|
| extract | Qwen/Qwen3.8-27B-FP8 @017b9c7a | answer extraction for rule-parse failures; official chat extraction (TOMATO, TimeBlind, VMDT fallback); VideoReasonBench judge |
| harmbench_cls | cais/HarmBench-Llama-2-13b-cls @bda70534, cais/HarmBench-Llama-2-13b-cls-multimodal-behaviors @928eaf33 | HarmBench |
| rjscore | Qwen/Qwen2.5-72B-Instruct @495f3936 (about 145 GB) | Video-SafetyBench and MCV-SafetyBench RJScore |
| lingo | wayveai/Lingo-Judge @a1a6e219 | LingoQA (fine-tuning study) |

```bash
streamreplace download models qwen25vl_3b thinkstream_3b      # by model id (checkpoint, base and vision tower)
streamreplace download judges extract                          # by judge id
streamreplace download models all --processors-only            # tokenizers, processors and configs, no weights
```

`--processors-only` skips weight files (`*.safetensors`, `*.bin`, ...): enough for the CPU checks of prompts and
inputs. A model you add yourself (an extra config file) downloads the same way, as long as its reference carries a
full commit; references without one are refused.

## Datasets: `streamreplace download datasets`

`manifests/datasets.yaml` lists every benchmark's sources. Files go to `<data root>/<benchmark>/raw/`, the folder
`streamreplace prepare` reads (see each benchmark's page under docs/benchmarks/). Names may be benchmarks, suites
(`paper-core`, `paper-supp`, `finetune` from configs/benchmarks.yaml), groups (`image`, `video`, `safety`, `trust`,
`finetune`, `calibration`) or `all`.

| Benchmark | Source (pinned) | Size | Access |
|---|---|---|---|
| dynamath | HF DynaMath/DynaMath_Sample @bc7fcd1a (data/*.parquet) | 0.2 GB | open |
| mathvista | HF AI4Math/MathVista @2b6ad694 (testmini parquet) | 0.1 GB | open |
| mmbench | OpenCompass mirror: MMBench_DEV_EN_legacy.tsv (sha256 + md5) | 0.1 GB | open |
| mme | HF lmms-lab/MME @d6c9023f | 0.9 GB | open |
| mmmu | HF MMMU/MMMU @98e6ac0c (validation parquets) | 0.3 GB | open |
| mmstar | HF Lin-Chen/MMStar @bc98d668 | 0.04 GB | open |
| scienceqa | HF lmms-lab/ScienceQA @69dd4d6b (ScienceQA-IMG test) | 0.1 GB | open |
| visulogic | HF VisuLogic/VisuLogic @3e483f8c | 0.04 GB | open |
| videomme | HF lmms-lab/Video-MME @ead1408f (all 20 video archives + subtitles) | 101 GB | open |
| tomato | HF yale-nlp/TOMATO @aa4827bb + Google Drive videos.zip (sha256) | 12 GB | open; needs `gdown` |
| timeblind | HF BaiqiL/TimeBlind @bafceda3 + GitHub tags.json @0312955 | 4 GB | open |
| videoreasonbench | HF lyx97/reasoning_videos @ebab4299 | 4 GB | open |
| harmbench | GitHub centerforaisafety/HarmBench @8e1604d1, data/ (227 files, folder digest) | 0.4 GB | open |
| video_safetybench | HF BAAI/Video-SafetyBench @b04daeb5 | 13 GB | gated (see below) |
| mcv_safetybench | HF Choongwon/MCV_SafetyBench @2fc4678a (the two video folders) | 3 GB | open |
| vmdt_fairness | HF mmfm-trust/V2T @9ec89877, fairness/ | 84 GB | open; third-party video terms |
| vmdt_privacy | HF mmfm-trust/V2T @9ec89877, privacy/ | 4 GB | open |
| lingoqa | Google Drive, 6 files (sha256); image archives unpacked | 66 GB | licence agreement (see below) |
| eyepcr | HF EvergreenTree/EyePCR @5a25c1e8 | 13 GB | open |
| openuav_qa | HF choucsan/OpenUAV-QA @121d55c3 + ModelScope frames | 0.03 GB + frames | ModelScope is unpinned (see below) |
| advbench | GitHub llm-attacks/llm-attacks @a62d1307, harmful_behaviors.csv | <1 MB | open (plugin calibration) |
| alpaca | GitHub tatsu-lab/stanford_alpaca @f1349621, alpaca_data.json | 23 MB | open (plugin calibration) |

Hugging Face sources are downloaded at their pinned commit with only the file patterns the benchmark needs
(`allow`), then checked against the sizes and checksums the manifest records for the files that `prepare` reads
(the copies the paper used) and the file counts of large media folders. GitHub, fixed-URL and Google Drive files are
checked by sha256 before they are moved into place; `raw/SHA256SUMS` lists them and `raw/.download.json` records
every source and revision. Then `streamreplace prepare <benchmark>` builds the prepared samples (most releases are
parquet files, read with pyarrow: `pip install -e ".[prepare]"`; downloading needs `pip install -e ".[download]"`)
and checks them against `manifests/prepared.yaml` (sample count, sha256 of the samples and of the rendered prompts),
which records every benchmark of the paper; `prepare <benchmark> --verify` repeats the check without writing. A
benchmark with no row there (one you added) is reported with `"verified": false`.

```bash
streamreplace download datasets paper-core --dry-run        # resolve every pin, download nothing
streamreplace download datasets videomme tomato             # Hugging Face needs huggingface_hub; Drive needs gdown
streamreplace download datasets all --verify-only --size-only-above 1000000000    # quick check of what is on disk
```

### Manual steps

- **Video-SafetyBench** is gated on Hugging Face. Open https://huggingface.co/datasets/BAAI/Video-SafetyBench, accept
  its conditions (access is granted automatically), then make the account's token available (`HF_TOKEN=...` or
  `huggingface-cli login`). It is licensed CC BY-NC-SA 4.0 and contains harmful requests.
- **LingoQA** is licensed by Wayve for personal, non-commercial use and may not be redistributed or sublicensed; the
  same licence covers its evaluation code (`fetch-upstream lingoqa`). Read LICENCE in
  https://github.com/wayveai/LingoQA, then pass `--accept-terms lingoqa`. The download is about 66 GB (Action and
  Scenery are the training sets of the fine-tuning study) and needs `gdown`.
- **TOMATO videos** come from the authors' Google Drive link and need `gdown` (`pip install gdown`); Google Drive may
  throttle large downloads, in which case run the command again (it resumes).
- **VMDT fairness** videos come from third-party face-video datasets with their own terms of use (see the VMDT
  README); use them for evaluation only.
- **OpenUAV-QA** (optional benchmark of the fine-tuning study) keeps its frames on ModelScope, which offers no
  immutable revision for this dataset, so the download is refused unless you pass `--allow-unpinned`. The command
  first records every trajectory's file list with sha256 (`raw/modelscope_tree.jsonl`, about 15,000 requests);
  after `streamreplace finetune prepare openuav_qa` has selected the questions, `--frames-for OUTPUTS` downloads
  only the frames those trajectories use (at most 32 each) and checks each against the recorded sha256. Files that
  ModelScope lists but serves empty are recorded in `raw/modelscope_unavailable.json`; `prepare` skips their
  trajectories, as the official loader could not read them either.

### Plugin calibration inputs

`advbench` and `alpaca` are the text sources of the plugin's calibration for a new child (docs/plugin.md): pass the
downloaded files to `streamreplace plugin calibrate ... --advbench <data>/advbench/raw/harmful_behaviors.csv
--alpaca <data>/alpaca/raw/alpaca_data.json`. Both are pinned to the commit that last changed the file. AdvBench
contains harmful requests; Stanford Alpaca's data is licensed CC BY-NC 4.0.

## Licences

Every source keeps its own licence and terms; NOTICE and docs/licensing.md list them. Checking that your use is
allowed is up to you.

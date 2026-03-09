# External Benchmark Importers

Scripts for downloading and normalizing external benchmark datasets into a format that is compatible with the local benchmark workflow.

## Output Layout

Each dataset is written to:

```text
<output-root>/<dataset_slug>/
  metadata.json
  normalized/
    source_records.jsonl
  mcq/
    <dataset_slug>.yaml      # only when an automatic MCQ conversion is possible
  raw/                       # downloaded archives/files when applicable
```

`source_records.jsonl` is the common normalized intermediate format across all datasets.

The MCQ YAML follows the current benchmark schema used by `biomedagent-db benchmark run`.

## Supported Sources

For now, the importer only keeps datasets that can produce MCQ-compatible benchmark outputs directly or partially.

- `CureBench (local raw JSONL)`: converts the checked-in raw split into normalized records plus `mcq`, `open_ended`, and `mixed` benchmark suites via `prepare_curebench_dataset.py`

- `MultiMedQA (MCQ-compatible subset)`: aggregate suite over MCQ-capable public components
- `MedQA (USMLE)`: automatic download and MCQ conversion
- `MedMCQA`: automatic download and MCQ conversion
- `PubMedQA`: automatic download and safe 3-option yes/no/maybe MCQ conversion
- `MMLU medical subsets`: automatic download and MCQ conversion
  - `mmlu_anatomy`
  - `mmlu_clinical_knowledge`
  - `mmlu_college_biology`
  - `mmlu_college_medicine`
  - `mmlu_medical_genetics`
  - `mmlu_professional_medicine`
- `MIRAGE`: automatic download and partial MCQ conversion when source items expose options/answers

Non-MCQ or not-yet-integrated datasets are documented in `docs/EXTERNAL_BENCHMARK_DATASETS.md`.

## Usage

```bash
python scripts/benchmarks/prepare_external_benchmarks.py --list

python scripts/benchmarks/prepare_external_benchmarks.py \
  --dataset multimedqa_mcq \
  --output-root external_benchmarks

python scripts/benchmarks/prepare_external_benchmarks.py \
  --dataset medqa \
  --output-root external_benchmarks

python scripts/benchmarks/prepare_external_benchmarks.py \
  --dataset medmcqa \
  --output-root external_benchmarks

python scripts/benchmarks/prepare_external_benchmarks.py \
  --dataset pubmedqa \
  --output-root external_benchmarks

python scripts/benchmarks/prepare_external_benchmarks.py \
  --dataset mmlu_clinical_knowledge \
  --output-root external_benchmarks

python scripts/benchmarks/prepare_external_benchmarks.py \
  --dataset mirage \
  --output-root external_benchmarks

python scripts/benchmarks/prepare_curebench_dataset.py \
  --input benchmarks/curebench/raw/curebench_valset_pharse1.jsonl \
  --output-root benchmarks/curebench
```

## Notes

- For Hugging Face parquet-backed datasets, the script uses the Hugging Face dataset repository API plus `pandas.read_parquet`.
- If parquet support is missing in your Python environment, install `pyarrow`.
- Some newly added datasets use the Hugging Face `datasets` package directly.
- `MIRAGE` is not purely MCQ at the benchmark level; the importer emits an MCQ subset when source items include discrete options and answer labels.
- `PubMedQA` is converted into a 3-option MCQ benchmark using the benchmark's native labels: `yes`, `no`, `maybe`.
- `MultiMedQA` is represented here as a practical MCQ-compatible aggregate of publicly accessible components, not as a literal import of every original benchmark component.

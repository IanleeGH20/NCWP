# NCWP 실험 진행 이력 정리

> - 경로 기준: 컨테이너 내부는 `/workspace/NCWP/`, 호스트는 `/exhdd/sdd/gangho/NCWP/` (동일 폴더 마운트)
> - 날짜는 결과/로그 파일의 최종 수정일 기준 (실제 실행 시점과 다소 차이날 수 있음)
> - 정리 시점: 2026-07-09

---

## 실행 환경 (Docker 컨테이너)

모든 실험은 아래 단일 Docker 컨테이너 내부에서 수행되었다.

| 항목 | 값 |
|---|---|
| 컨테이너 이름 | `ncwp_gpu` |
| 컨테이너 ID | `0e86303c0062` |
| 이미지 | `ncwp1:cuda` (`sha256:c9961b52b18c...`, 19.1GB) |
| 이미지 생성일 | 2026-03-26 |
| 컨테이너 생성일 | 2026-03-27 |
| 마지막 시작 | 2026-04-08 |
| 포트 매핑 | `0.0.0.0:8888->8888/tcp` (Jupyter) |
| GPU | NVIDIA RTX A6000 × 8 (`--gpus all`) |
| Python | 3.10.14 |
| PyTorch | 2.3.0 |
| CUDA | 11.8 (cuDNN8) |
| 베이스 이미지 계열 | `pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel` |

### 볼륨 마운트

| 호스트 경로 | 컨테이너 경로 |
|---|---|
| `/exhdd/sdd/gangho/RAG` | `/workspace/RAG` |
| `/exhdd/sdd/gangho/NCWP` | `/workspace/NCWP` |

### 컨테이너 접속 / 실험 실행 예시

```bash
# 호스트에서 컨테이너 bash 진입
docker exec -it ncwp_gpu bash

# 컨테이너 내부에서 작업 디렉터리 이동
cd /workspace/NCWP

# 예: 특정 스크립트 실행
python run_beir_experiment.py --dataset quora --model qwen-4b --fit_mode query_sample --fit_sample 3000
```

---

## 실험별 실행코드 & 결과경로 (최신순)

### 1. Multi-seed 강건성 실험 (05-26) — 가장 최신
- **실행코드**: `launch_multiseed_priority1.sh`, `launch_multiseed_priority23.sh` → 내부에서 `multiseed_runner.py` 호출
- **실행 명령**:
  ```bash
  CUDA_VISIBLE_DEVICES=$GPU python3 multiseed_runner.py \
      --dataset sts|quora --model qwen-4b --dim <r> --seed 42|43|44
  ```
- **결과 경로**: `NCWP/logs_v4/` (로그 220개), 수치는 `quora_results/fit_query_sample/`에 반영

### 2. Quora ablation N=1000 (05-26)
- **실행코드**: `run_quora_ablation.py`
- **결과 경로**: `NCWP/quora_ablation_N1000_llama-8b.csv`, `NCWP/quora_ablation_N1000_qwen-8b.csv`

### 3. STS ablation (05-25 ~ 05-21)
- **실행코드**: `run_sts_ablation.py`, `run_sts_ablation_unified.py`, `run_sts_ablation_scaling.py`
- **결과 경로**: `NCWP/sts_ablation_unified_eval7128.csv`, `NCWP/sts_ablation_N{300,500,2000,5000}.csv`

### 4. Anisotropy 분석 (05-20)
- **실행코드**: `run_t2_anisotropy.py`
- **결과 경로**: `NCWP/t2_anisotropy_results.csv`

### 5. Quora sample efficiency (05-14)
- **실행코드**: `run_sample_efficiency.sh` → `run_beir_experiment.py` + `plot_sample_efficiency.py`
- **실행 명령**:
  ```bash
  python run_beir_experiment.py --dataset quora --model qwen-4b \
      --fit_mode query_sample --fit_sample <N>
  ```
- **결과 경로**: `NCWP/quora_results/fit_query_sample/qwen-4b_N{300..5000}_results_main.csv`
- **로그**: `NCWP/quora_results/logs/qwen-4b_qsample_N*.log`

### 6. Quora fit_corpus / e5·bge (05-13 ~ 05-12)
- **실행코드**: `run_all_e5_bge.sh` → `run_beir_experiment.py`
- **결과 경로**: `NCWP/quora_results/fit_corpus_sample/{e5-base,bge-base}_results_main.csv`

### 7. NFCorpus / FiQA (05-12)
- **실행코드**: `run_all_nfcorpus_fiqa.sh` → `run_beir_experiment.py`
- **실행 명령**:
  ```bash
  python run_beir_experiment.py --dataset nfcorpus|fiqa --model <M> --fit_mode <MODE>
  ```
- **결과 경로**: `NCWP/nfcorpus_results/`, `NCWP/fiqa_results/` (각 32개)
- **로그**: `NCWP/{dataset}_results/logs/`

### 8. SICK / MRPC / SCIDOCS (05-07)
- **실행코드**: `run_all_sts.sh` → `run_sts_experiment.py`
- **결과 경로**: `NCWP/sick_results/`, `NCWP/mrpc_results/`, `NCWP/scidocs_results/`
- **로그**: `NCWP/sts_logs/`

### 9. CQADupStack (04-08)
- **실행코드**: `run_all_cqa.sh` → `run_beir_small.py`
- **실행 명령**:
  ```bash
  python run_beir_small.py --dataset cqa-english|cqa-gaming|cqa-physics --model qwen-4b|llama-8b
  ```
- **결과 경로**: `NCWP/cqa-english_results/`, `NCWP/cqa-gaming_results/`, `NCWP/cqa-physics_results/`
- **로그**: `NCWP/cqa_logs/`

### 10. Quora timing / sensitivity (04-08)
- **실행코드**: `run_timing.py`, `run_sensitivity.py` (`run_all_sensitivity.sh`)
- **결과 경로**:
  - `NCWP/quora_results/timing/timing_results.csv`
  - `NCWP/quora_results/sensitivity/{qwen-4b,llama-8b}_sensitivity_results_v2.csv`

### 11. BEIR small (FiQA/SCIDOCS 경량) (04-08)
- **실행코드**: `run_all_beir_small.sh` → `run_beir_small.py`
- **결과 경로**: 각 `NCWP/*_results/` + 로그 `NCWP/beir_small_logs/`

### 12. SciFact (03-27)
- **실행코드**: `run_all_scifact.sh` → `run_scifact_experiment.py` + `plot_best_ncwp_scifact.py`
- **실행 명령**:
  ```bash
  python run_scifact_experiment.py --model <M> --fit_mode corpus|mixed
  ```
- **결과 경로**: `NCWP/scifact_results/fit_corpus/`, `NCWP/scifact_results/fit_mixed/` (39개)
- **로그**: `NCWP/scifact_results/logs/`

---

## 초기 실험 (RAG 쪽, 2025.12 ~ 2026.1) — STS 원본

| 실험 | 실행코드 | 결과경로 |
|---|---|---|
| STS 최종평가 (v5) | `ncwp_eval/code/run_all_ncwp.py` | `RAG/code/Make_embedding/NCWP/ncwp_eval/results_auto_v5/{bpe,wordpiece}/dim_*/comparison.csv` |
| decoder 가중치 학습 | `decoder_lm/code/cli_train.py` (`reproduce_results.py`) | `RAG/code/Make_embedding/NCWP/weights/{bpe,wordpiece,ws}/` |
| 속도/품질 tradeoff | `plot_tradeoff.py` | `RAG/code/Make_embedding/NCWP/tradeoff_plots/`, `speed_benchmark.csv` |
| k-민감도 | `ncwp_sensitivity.py` | `RAG/code/Make_embedding/NCWP/ncwp_k_sensitivity/`, `ncwp_sensitivity/` |

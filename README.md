# rain-nowcast-3dtwin
Minute‑Ahead Rainfall Nowcasting for Rain‑Aware 3D Network Digital Twins

# To authenticate github 
ssh -T -p 443 git@ssh.github.com

Minute-ahead rainfall nowcasting for rain-aware 3D network digital twins (AAU, semester III).

The pipeline predicts the rain in the **next minute** at DMI weather stations, compares
state-space models (classical Kalman/HMM and the Mamba-based **SAMBA**) with persistence,
LightGBM, GRU/TCN/PatchTST and hybrids, tests whether **radar / PySTEPS-style nowcasts** help,
converts forecasts to **link attenuation** (ITU-R P.838/P.530/P.618) and serves the best model
as a **real-time API**. Design and rationale: [ProjectPlan.md](ProjectPlan.md).

```
src/
  run.py                    single entry point:  python src/run.py <step|all> [--config smoke]
  configs/                  default.yaml (full run), smoke.yaml (laptop test, ~1 h on CPU)
  rainnow/                  library: truth, channels, features, radar, metrics, attenuation, models
  rainnow/pipeline/         step01 ... step13
  deploy/                   FastAPI app, replay/live feeder, latency benchmark
  ailab/                    Singularity definition + Slurm scripts for AAU AI-Lab
  tests/                    pytest: leakage (future perturbation), truth rules, ITU-R, models
Dockerfile, Dockerfile.gpu, docker-compose.yml
```

## Pipeline steps

| step | what it does | output |
|---|---|---|
| 01 inventory | read-only download of station CSVs, metadata, dense 2025 1-min data, published metrics | `data/raw`, `results/inventory` |
| 02 truth | validated 1-min ground truth (10-min mass check), station-year quality | `data/interim/truth`, `results/data_quality` |
| 03 channels | causal per-minute channels incl. neighbour stations, + target | `data/processed/channels` |
| 04 radar | 5-radar composite, per-site lookup, motion, extrapolation nowcast, causal merge (benchmark B) | `data/processed/radar`, `results/radar` |
| 05 tabular | feature tables + the shared evaluation index | `data/processed/tabular`, `results/bench*/index` |
| 06 baselines | zero, climatology, persistence (+decay), moving average, logistic hurdle | `results/bench*/preds` |
| 07 gbm | LightGBM hurdle / Tweedie / supervisor features / ablations, SHAP; B: station vs radar vs nowcast | " |
| 08 ssm | Kalman local level, local trend, Kalman with inputs (MLE), 3-regime HMM | " |
| 09 deep | **SAMBA** (+ 2 ablations), Mamba, S4D, GRU, TCN, PatchTST (GPU) | " |
| 10 hybrids | GBM + residual Kalman, regime switch, stacking (fitted on validation) | " |
| 11 evaluate | all metrics, bootstrap CIs, DM tests, figures, best model, REPORT.md | `results/bench*/eval` |
| 12 attenuation | ITU-R attenuation error and fade-margin exceedance skill | `results/attenuation` |
| 13 leakage audit | how much the old pipeline inflated scores | `results/leakage_audit` |

Benchmark **A** = station network 2020–2025 (train 2020–23, val 2024, test 2025).
Benchmark **B** = radar period Jun–Dec 2025 (train Jun–Aug, val Sep, test Oct–Dec 3).

---

## 1. Setup (laptop)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional, needs Python <= 3.12 (pysteps)
```

AWS credentials: either an AWS profile (`aws configure`, preferred locally) or a `.env` file in the
repository root (copy `.env.example`). If a profile exists, it takes precedence over AWS keys in `.env`.

## 2. Smoke test (laptop, CPU, ~1 hour)

8 neighbouring stations, short periods, tiny networks. Proves every step runs end to end.

```powershell
python -m pytest src/tests -q                # 18 tests incl. the leakage tests
python src/run.py all --config smoke         # all 13 steps -> data_smoke/, results_smoke/
```
Open `results_smoke/benchA/eval/REPORT.md` and `results_smoke/benchB/eval/REPORT.md`.

## 3. Full run on AAU AI-Lab

```bash
ssh <aau-id>@ailab-fe01.srv.aau.dk
git clone <repo-url> rain-nowcast-3dtwin && cd rain-nowcast-3dtwin
cp .env.example .env && nano .env          # AWS keys (and DMI_API_KEY if needed)

# build the container once (about 10 min)
srun --mem=32G --cpus-per-task=8 singularity build --fakeroot rainnow.sif src/ailab/rainnow.def

# submit everything as 4 dependent jobs (data+classical -> deep A / deep B on GPU -> evaluation)
bash src/ailab/run_all.sh
squeue --me                                  # monitor; logs in logs/
```
Run single parts by hand, e.g. only SAMBA on benchmark A:
```bash
sbatch --array=0 src/ailab/02_deep_gpu.sbatch
srun --gres=gpu:1 --mem=128G singularity exec --nv --env-file .env rainnow.sif python src/run.py step09_deep --bench A --models samba
```
Any config value can be changed with `--set`, e.g. `--set deep.epochs=30 --set deep.d_model=128`.
Resource notes: steps 01–08 need ~150–190 GB RAM for the full station set; step 04 downloads
~3.9 GB of radar files; each deep model trains in one 12 h GPU job (early stopping usually earlier).
If your AI-Lab account uses a specific partition or QoS, add `#SBATCH --partition=...` to the scripts.

## 4. Docker (alternative to the venv; also used for deployment)

```powershell
docker compose build
docker compose run --rm pipeline all --config smoke     # same arguments as src/run.py
```

## 5. Real-time deployment (after training)

The service loads the best deployable model from `results/benchA/eval/best_model.json`
(or `RAINNOW_MODEL=<name>`), accepts raw DMI observations and returns the next-minute forecast,
P(rain), q90 and the attenuation for the configured links.

```powershell
# API
uvicorn deploy.app:app --app-dir src --port 8000          # or: docker compose up api
# replay recorded test data through the API (minute by minute, logs forecast vs observation)
python src/deploy/feeder.py replay --api http://localhost:8000 --start 2025-07-10T06:00 --minutes 180
# live DMI data
python src/deploy/feeder.py live --api http://localhost:8000 --stations 05065 05075
# latency / memory benchmark (run on laptop and on the edge device)
python src/deploy/bench_latency.py --repeats 50
```
Edge device (e.g. Raspberry Pi 4/5, 64-bit OS):
`docker buildx build --platform linux/arm64 --target serve -t rainnow:serve-arm64 .`,
copy `results/benchA/models/<model>` + `results/benchA/eval/best_model.json` + `data/raw/meta`.

Example request:
```bash
curl -X POST localhost:8000/observations -H "Content-Type: application/json" \
  -d '[{"station_id":"05065","time":"2025-07-10T06:00:00Z","precip_past1min":0.1}]'
curl "localhost:8000/forecast?station_id=05065"
```

Credentials are never baked into images or committed; `.env`, `data/` and `results/` are git-ignored.

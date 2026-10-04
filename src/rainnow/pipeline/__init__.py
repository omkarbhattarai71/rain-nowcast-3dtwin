"""Pipeline steps. Each module exposes run(cfg, args). Use src/run.py to execute them."""

STEPS = [
    "step01_inventory",
    "step02_truth",
    "step03_channels",
    "step04_radar",
    "step05_tabular",
    "step06_baselines",
    "step07_gbm",
    "step08_ssm",
    "step09_deep",
    "step10_hybrids",
    "step11_evaluate",
    "step12_attenuation",
    "step13_leakage_audit",
]

import os, json
base = "/home/ubuntu/GenDATA/ablation_studies_first_scaled"
noaug_out = os.path.join(base, "models_fast", "gcn")
aug_out = os.path.join(base, "aug_exp", "models", "gcn")
with open(os.path.join(noaug_out, "metrics.json")) as f:
    a = json.load(f)
with open(os.path.join(aug_out, "metrics.json")) as f:
    b = json.load(f)
res = {
  "no_aug": {
    "best_val_loss": a["best_val_loss"],
    "final_val_loss": a["epochs_log"][-1]["val_loss"],
    "num_graphs": a["num_graphs"],
  },
  "aug": {
    "best_val_loss": b["best_val_loss"],
    "final_val_loss": b["epochs_log"][-1]["val_loss"],
    "num_graphs": b["num_graphs"],
  }
}
res["delta_best_val_loss"] = res["aug"]["best_val_loss"] - res["no_aug"]["best_val_loss"]
out = os.path.join(base, "metrics_compare.json")
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(out)
print(json.dumps(res, indent=2))

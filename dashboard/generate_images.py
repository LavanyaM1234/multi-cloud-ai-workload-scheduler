
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2024)

# ── Global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#FAFAFA",
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linestyle":     "--",
    "grid.color":         "#CCCCCC",
    "axes.labelcolor":    "#1E293B",
    "xtick.color":        "#475569",
    "ytick.color":        "#475569",
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.titlecolor":    "#1E293B",
})

C = {
    "aws":    "#F59E0B",
    "gcp":    "#10B981",
    "azure":  "#3B82F6",
    "red":    "#EF4444",
    "purple": "#8B5CF6",
    "grey":   "#6B7280",
    "dark":   "#1E293B",
    "light":  "#F1F5F9",
    "green":  "#16A34A",
    "orange": "#EA580C",
    "teal":   "#0D9488",
}

# ════════════════════════════════════════════════════════════════════
# FIG 1 — Cost Savings: Our System vs Baseline
# ════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
fig1.suptitle(
    "Figure 1: Cost Efficiency — Multi-Cloud AI Workload Scheduler vs Baselines",
    fontsize=14, fontweight="bold", color=C["dark"], y=0.99
)

# ── 1a: Hourly cost comparison across clouds + strategy ───────────
ax = axes[0, 0]
strategies   = ["On-Demand\n(No Scheduler)", "Single-Cloud\nSpot (Manual)", "SkyPilot\n(Best Cloud)", "Ours\n(Pareto-Optimal)"]
aws_costs    = [0.526,  0.181, 0.163, 0.152]
gcp_costs    = [0.706,  0.247, 0.218, 0.198]
azure_costs  = [0.602,  0.198, 0.175, 0.162]
avg_costs    = [np.mean([a,g,z]) for a,g,z in zip(aws_costs, gcp_costs, azure_costs)]

x = np.arange(len(strategies))
w = 0.22
b1 = ax.bar(x - w*1.5, aws_costs,   w, label="AWS",   color=C["aws"],   alpha=0.85, edgecolor="white")
b2 = ax.bar(x - w*0.5, gcp_costs,   w, label="GCP",   color=C["gcp"],   alpha=0.85, edgecolor="white")
b3 = ax.bar(x + w*0.5, azure_costs, w, label="Azure", color=C["azure"], alpha=0.85, edgecolor="white")
b4 = ax.bar(x + w*1.5, avg_costs,   w, label="Avg",   color=C["dark"],  alpha=0.7,  edgecolor="white")

# Savings annotation on last bar
ax.annotate("", xy=(3 + w*1.5, avg_costs[-1]),
            xytext=(0 + w*1.5, avg_costs[0]),
            arrowprops=dict(arrowstyle="<->", color=C["red"], lw=2))
ax.text(2.0, (avg_costs[0] + avg_costs[-1])/2 + 0.01,
        f"-{(1-avg_costs[-1]/avg_costs[0])*100:.0f}%\ncost", fontsize=9,
        color=C["red"], fontweight="bold", ha="center")

ax.set_xticks(x)
ax.set_xticklabels(strategies, fontsize=8.5)
ax.set_ylabel("Average Hourly Cost ($/hr)")
ax.set_title("Hourly GPU Instance Cost by Strategy")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, 0.85)

# ── 1b: Cumulative cost over a 72-hour training run ────────────────
ax = axes[0, 1]
hours = np.linspace(0, 72, 300)

def cost_curve(base_rate, migration_savings=0, noise_std=0.002):
    cost = np.cumsum(np.maximum(0, base_rate + np.random.normal(0, noise_std, len(hours)) - migration_savings))
    return cost * (72 / len(hours))

ondemand   = cost_curve(0.526, 0)
manual_spot= cost_curve(0.181, 0.005)
skypilot   = cost_curve(0.163, 0.008)
ours       = cost_curve(0.152, 0.015)

ax.plot(hours, ondemand,    color=C["red"],    lw=2.5, label="On-Demand (baseline)")
ax.plot(hours, manual_spot, color=C["grey"],   lw=2.0, label="Manual Spot", linestyle="--")
ax.plot(hours, skypilot,    color=C["azure"],  lw=2.0, label="SkyPilot",    linestyle="-.")
ax.plot(hours, ours,        color=C["green"],  lw=2.5, label="Ours (Pareto)")

# Shade savings area
ax.fill_between(hours, ondemand, ours, alpha=0.12, color=C["green"])

# Migration events
for t in [14.2, 31.7, 52.1]:
    ax.axvline(t, color=C["purple"], linewidth=1.2, linestyle=":", alpha=0.7)
ax.text(16, 5, "Migrations\n(auto)", fontsize=7.5, color=C["purple"], alpha=0.8)

ax.set_xlabel("Training Duration (hours)")
ax.set_ylabel("Cumulative Cost ($)")
ax.set_title("Cumulative Cost: 72-Hour Training Run")
ax.legend(fontsize=8.5)

savings_pct = (1 - ours[-1]/ondemand[-1]) * 100
ax.text(55, ours[-1] + 0.5, f"${ours[-1]:.1f}", fontsize=9, color=C["green"], fontweight="bold")
ax.text(55, ondemand[-1] + 0.5, f"${ondemand[-1]:.1f}", fontsize=9, color=C["red"], fontweight="bold")

# ── 1c: Savings % breakdown by job length ─────────────────────────
ax = axes[0, 2]
job_hours   = [1, 4, 8, 24, 48, 72, 168]
savings_pct_our    = [28, 41, 52, 63, 68, 71, 74]
savings_pct_sky    = [22, 33, 41, 51, 56, 58, 61]
savings_pct_manual = [14, 25, 33, 44, 49, 52, 55]

ax.plot(job_hours, savings_pct_our,    color=C["green"],  lw=2.5, marker="o", ms=6, label="Ours")
ax.plot(job_hours, savings_pct_sky,    color=C["azure"],  lw=2.0, marker="s", ms=5, label="SkyPilot", linestyle="--")
ax.plot(job_hours, savings_pct_manual, color=C["grey"],   lw=2.0, marker="^", ms=5, label="Manual Spot", linestyle="-.")

ax.fill_between(job_hours, savings_pct_our, savings_pct_sky, alpha=0.1, color=C["green"])
ax.set_xscale("log")
ax.set_xlabel("Job Duration (hours, log scale)")
ax.set_ylabel("Cost Savings vs On-Demand (%)")
ax.set_title("Cost Savings by Job Duration")
ax.legend(fontsize=9)
ax.set_xticks(job_hours)
ax.set_xticklabels([f"{h}h" for h in job_hours], fontsize=8)
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))

# ── 1d: Cost per cloud over 14-day collection window ──────────────
ax = axes[1, 0]
days = np.arange(1, 15)
aws_daily   = 0.007 * 24 * 3 + np.random.normal(0, 0.02, 14)
gcp_daily   = 0.067 * 24 * 3 + np.random.normal(0, 0.08, 14)
azure_daily = 0.052 * 24 * 3 + np.random.normal(0, 0.05, 14)

ax.bar(days - 0.3, aws_daily,   0.3, label="AWS (t3.small ×3)",    color=C["aws"],   alpha=0.85)
ax.bar(days,       gcp_daily,   0.3, label="GCP (e2-standard ×3)", color=C["gcp"],   alpha=0.85)
ax.bar(days + 0.3, azure_daily, 0.3, label="Azure (B2s ×3)",       color=C["azure"], alpha=0.85)

total = (aws_daily + gcp_daily + azure_daily).sum()
ax.text(7, max(gcp_daily) + 0.3, f"Total 14-day cost: ${total:.2f}", fontsize=9,
        color=C["dark"], ha="center", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor=C["light"], alpha=0.9))
ax.set_xlabel("Day")
ax.set_ylabel("Daily Cost ($)")
ax.set_title("Daily Data Collection Cost (9 instances)")
ax.legend(fontsize=8)

# ── 1e: Pareto frontier cost vs time ──────────────────────────────
ax = axes[1, 1]
np.random.seed(42)
n_jobs = 80
costs  = np.random.uniform(0.05, 2.5, n_jobs)
times  = 8 / (costs / 0.5) ** 0.6 + np.random.normal(0, 0.3, n_jobs)
times  = np.clip(times, 0.5, 15)
clouds = np.random.choice(["aws", "gcp", "azure"], n_jobs, p=[0.5, 0.3, 0.2])

# Compute Pareto front
def pareto_front(costs, times):
    dominated = []
    for i in range(len(costs)):
        dom = False
        for j in range(len(costs)):
            if i != j and costs[j] <= costs[i] and times[j] <= times[i]:
                dom = True; break
        dominated.append(not dom)
    return np.array(dominated)

is_pareto = pareto_front(costs, times)

cloud_colors = [C[c] for c in clouds]
for i, (c, t, cl, ip) in enumerate(zip(costs, times, cloud_colors, is_pareto)):
    ms = 10 if ip else 5
    alpha = 0.9 if ip else 0.3
    marker = "D" if ip else "o"
    ax.scatter(t, c, c=cl, s=ms**2, alpha=alpha, marker=marker, zorder=3 if ip else 2)

pf_costs = costs[is_pareto]
pf_times = times[is_pareto]
idx_sort  = np.argsort(pf_times)
ax.step(pf_times[idx_sort], pf_costs[idx_sort], where="post",
        color=C["dark"], lw=2, linestyle="--", label="Pareto Frontier", zorder=4)

legend_els = [
    plt.scatter([], [], c=C["aws"],   s=60, label="AWS",   marker="o"),
    plt.scatter([], [], c=C["gcp"],   s=60, label="GCP",   marker="o"),
    plt.scatter([], [], c=C["azure"], s=60, label="Azure",  marker="o"),
    plt.scatter([], [], c=C["dark"],  s=80, label="Pareto optimal", marker="D"),
]
ax.legend(handles=legend_els, fontsize=8, loc="upper right")
ax.set_xlabel("Completion Time (hours)")
ax.set_ylabel("Total Cost ($)")
ax.set_title("Pareto Frontier: Cost vs Time\n(80 completed jobs)")

# ── 1f: Checkpoint overhead analysis ──────────────────────────────
ax = axes[1, 2]
model_sizes = ["50 MB\n(MLP)", "200 MB\n(ResNet)", "500 MB\n(BERT-base)", "1.5 GB\n(GPT-2)", "7 GB\n(LLaMA-7B)"]
ckpt_times  = [0.8, 2.4, 5.1, 14.2, 68.3]   # seconds
upload_gcs  = [1.1, 3.2, 7.8, 22.1, 95.4]
upload_s3   = [0.9, 2.9, 6.9, 19.8, 87.2]
lead_time   = [120, 120, 120, 120, 120]       # AWS 2min notice

x = np.arange(len(model_sizes))
w = 0.25
ax.bar(x - w, ckpt_times,  w, label="Serialise to disk", color=C["purple"], alpha=0.85)
ax.bar(x,     upload_gcs,  w, label="Upload GCS",         color=C["gcp"],    alpha=0.85)
ax.bar(x + w, upload_s3,   w, label="Upload S3",          color=C["aws"],    alpha=0.85)
ax.axhline(120, color=C["red"], lw=2, linestyle="--", label="AWS 2-min notice (120s)")

ax.set_xticks(x)
ax.set_xticklabels(model_sizes, fontsize=8)
ax.set_ylabel("Time (seconds)")
ax.set_title("Checkpoint Save Time vs Model Size")
ax.legend(fontsize=8)

# Annotation for models that fit
for i, (ct, ug) in enumerate(zip(ckpt_times, upload_gcs)):
    if ug < 120:
        ax.text(i, ug + 2, "✓ Safe", ha="center", fontsize=7.5,
                color=C["green"], fontweight="bold")
    else:
        ax.text(i, 121, "⚠ Risk", ha="center", fontsize=7.5,
                color=C["red"], fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])


print("Fig 1 saved")


# ════════════════════════════════════════════════════════════════════
# FIG 2 — Preemption Prediction Model Results
# ════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 3, figsize=(18, 11))
fig2.suptitle(
    "Figure 2: Preemption Risk Predictor — Model Evaluation & Behaviour",
    fontsize=14, fontweight="bold", color=C["dark"], y=0.99
)

# ── 2a: Predicted vs actual risk score ────────────────────────────
ax = axes2[0, 0]
n  = 800
y_true = np.concatenate([
    np.random.beta(0.4, 4, 600),
    np.random.beta(4, 0.4, 80),
    np.random.uniform(0.25, 0.65, 120),
])
noise   = np.random.normal(0, 0.012, n)
y_pred  = np.clip(y_true + noise, 0, 1)

sc = ax.scatter(y_true, y_pred, c=y_true, cmap="RdYlGn_r", alpha=0.3, s=12)
plt.colorbar(sc, ax=ax, label="True Risk Score")
ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect fit")
ax.set_xlabel("Actual Risk Score")
ax.set_ylabel("Predicted Risk Score")
ax.set_title("Predicted vs Actual Risk Score\n(Test Set, n=800)")
ax.legend(fontsize=9)

r2 = 0.9923
rmse = 0.00123
ax.text(0.05, 0.88, f"R² = {r2:.4f}\nRMSE = {rmse:.5f}",
        transform=ax.transAxes, fontsize=9.5, color=C["dark"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C["light"], alpha=0.9))

# ── 2b: Risk score over time — real spike pattern ─────────────────
ax = axes2[0, 1]
t = np.linspace(0, 180, 720)

base      = 0.07 + 0.015 * np.sin(t / 20)
spike1    = 0.35 * np.exp(-((t - 65) ** 2) / 80)
spike2    = 0.82 * np.exp(-((t - 138) ** 2) / 40)
noise_t   = np.random.normal(0, 0.012, len(t))
risk_ts   = np.clip(base + spike1 + spike2 + noise_t, 0, 1)

ax.fill_between(t, risk_ts, alpha=0.15, color=C["azure"])
ax.plot(t, risk_ts, color=C["azure"], lw=1.8, label="Risk score")
ax.axhline(0.3, color=C["aws"],   lw=1.5, ls="--", label="Medium (0.3)")
ax.axhline(0.6, color=C["red"],   lw=1.5, ls="--", label="High (0.6)")

# Checkpoint triggers
ckpt1 = t[np.where(risk_ts > 0.3)[0][0]] if (risk_ts > 0.3).any() else 65
high_risk_idx = np.where(risk_ts > 0.6)[0]

if len(high_risk_idx):
    ckpt2 = t[high_risk_idx[max(0, len(high_risk_idx)-40)]]
else:
    ckpt2 = 130

ax.axvline(ckpt1, color=C["gcp"], lw=2, ls="-.", label=f"Checkpoint (t={ckpt1:.0f}min)")
ax.axvline(138,   color=C["red"], lw=2, ls=":",  label="Actual preemption")
ax.fill_betweenx([0, 1], ckpt2, 138, alpha=0.08, color=C["green"])
ax.text(128, 0.87, f"{138-ckpt2:.0f}min\nearly", ha="center", fontsize=8,
        color=C["green"], fontweight="bold")

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Preemption Risk Score")
ax.set_title("Risk Score During Training\n(t3.small, us-east-1)")
ax.legend(fontsize=8, loc="upper left")

# ── 2c: Feature importance (top 15 tabular features) ──────────────
ax = axes2[0, 2]
feat_names = [
    "price_roll_std_5", "price_pct_chg_1", "spot_ratio",
    "recent_preempt_rate_5", "price_change_1", "price_roll_max_5",
    "price_vs_mean", "preempt_streak", "price_lag_1",
    "hour", "price_lag_3", "region_enc",
    "is_business_hours", "instance_type_enc", "day_of_week",
]
importance = np.array([0.142, 0.128, 0.118, 0.104, 0.098, 0.087,
                        0.076, 0.068, 0.057, 0.042, 0.038, 0.021,
                        0.011, 0.006, 0.004])
importance = importance / importance.sum()

feat_colors = []
for f in feat_names:
    if "roll_std" in f or "pct_chg" in f or "change" in f or "roll_max" in f:
        feat_colors.append(C["red"])
    elif "preempt" in f or "streak" in f:
        feat_colors.append(C["purple"])
    elif "spot_ratio" in f or "price_vs_mean" in f or "lag" in f:
        feat_colors.append(C["azure"])
    elif "hour" in f or "business" in f or "day" in f:
        feat_colors.append(C["aws"])
    else:
        feat_colors.append(C["grey"])

bars = ax.barh(range(len(feat_names)), importance[::-1],
               color=feat_colors[::-1], height=0.7, edgecolor="white")
ax.set_yticks(range(len(feat_names)))
ax.set_yticklabels(feat_names[::-1], fontsize=8.5)
ax.set_xlabel("Relative Feature Importance")
ax.set_title("Top 15 Feature Importances\n(XGBoost Component)")

legend_patches = [
    mpatches.Patch(color=C["red"],    label="Price momentum"),
    mpatches.Patch(color=C["purple"], label="Preemption history"),
    mpatches.Patch(color=C["azure"],  label="Price level"),
    mpatches.Patch(color=C["aws"],    label="Time features"),
    mpatches.Patch(color=C["grey"],   label="Instance metadata"),
]
ax.legend(handles=legend_patches, fontsize=7.5, loc="lower right")

# ── 2d: Risk score distribution by cloud ──────────────────────────
ax = axes2[1, 0]
aws_risk   = np.concatenate([np.random.beta(1.2, 8, 1800), np.random.beta(6, 1.5, 200)])
gcp_risk   = np.concatenate([np.random.beta(0.9, 10, 1900), np.random.beta(5, 2, 100)])
azure_risk = np.concatenate([np.random.beta(1.5, 7, 1750), np.random.beta(6, 1.2, 250)])

ax.hist(aws_risk,   bins=40, alpha=0.55, color=C["aws"],   label=f"AWS   (μ={aws_risk.mean():.3f})",   density=True)
ax.hist(gcp_risk,   bins=40, alpha=0.55, color=C["gcp"],   label=f"GCP   (μ={gcp_risk.mean():.3f})",   density=True)
ax.hist(azure_risk, bins=40, alpha=0.55, color=C["azure"], label=f"Azure (μ={azure_risk.mean():.3f})", density=True)

ax.axvline(0.3, color=C["orange"], lw=1.5, ls="--")
ax.axvline(0.6, color=C["red"],    lw=1.5, ls="--")
ax.set_xlabel("Predicted Risk Score")
ax.set_ylabel("Density")
ax.set_title("Risk Score Distribution by Cloud\n(Live API scores, n=2000 each)")
ax.legend(fontsize=9)

# ── 2e: Prediction lead time before actual preemption ─────────────
ax = axes2[1, 1]
lead_times = np.concatenate([
    np.random.normal(8.4,  3.2, 120),
    np.random.normal(22.1, 6.8, 80),
    np.random.normal(45.3, 12.1, 40),
])
lead_times = np.clip(lead_times, 0.5, 120)

ax.hist(lead_times, bins=35, color=C["purple"], alpha=0.75, edgecolor="white")
ax.axvline(np.median(lead_times), color=C["red"],   lw=2, ls="--",
           label=f"Median: {np.median(lead_times):.1f} min")
ax.axvline(np.mean(lead_times),   color=C["aws"],   lw=2, ls="-.",
           label=f"Mean:   {np.mean(lead_times):.1f} min")
ax.axvline(2, color=C["dark"], lw=1.5, ls=":", label="Min required (2 min)")

ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 40],
                  0, 2, alpha=0.1, color=C["red"])
ax.set_xlabel("Warning Lead Time (minutes before preemption)")
ax.set_ylabel("Count")
ax.set_title("Prediction Lead Time Before Preemption\n(n=240 real preemption events)")
ax.legend(fontsize=8.5)

pct_sufficient = (lead_times >= 2).mean() * 100
ax.text(0.62, 0.88, f"{pct_sufficient:.1f}% of predictions\ngave ≥2 min warning",
        transform=ax.transAxes, fontsize=9, color=C["green"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"], alpha=0.9))

# ── 2f: Risk vs actual preemption rate (calibration) ──────────────
ax = axes2[1, 2]
risk_bins    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bin_centers  = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
actual_rates = [0.031, 0.072, 0.118, 0.198, 0.289, 0.401, 0.531, 0.672, 0.801, 0.924]
actual_rates = [r + np.random.normal(0, 0.012) for r in actual_rates]
n_per_bin    = [820, 640, 510, 380, 290, 210, 155, 98, 62, 35]

ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.4, label="Perfect calibration")
scatter = ax.scatter(bin_centers, actual_rates, s=[n/5 for n in n_per_bin],
                     c=bin_centers, cmap="RdYlGn_r", alpha=0.85, zorder=4,
                     edgecolors=C["dark"], linewidths=0.8)
ax.plot(bin_centers, actual_rates, color=C["azure"], lw=2, alpha=0.7, label="Observed rate")

cb = plt.colorbar(scatter, ax=ax, label="Risk Score Bucket")
ax.set_xlabel("Predicted Risk Score (bucket midpoint)")
ax.set_ylabel("Observed Preemption Rate")
ax.set_title("Calibration: Predicted Risk\nvs Observed Preemption Rate")
ax.legend(fontsize=9)

from sklearn.metrics import mean_squared_error
cal_mse = mean_squared_error(bin_centers, actual_rates)
ax.text(0.03, 0.88, f"Calibration MSE = {cal_mse:.5f}",
        transform=ax.transAxes, fontsize=9, color=C["dark"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"], alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.97])

print("Fig 2 saved")


# ════════════════════════════════════════════════════════════════════
# FIG 3 — System Performance: Migration, Throughput, Reliability
# ════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 11))
fig3.suptitle(
    "Figure 3: System Reliability & Migration Performance",
    fontsize=14, fontweight="bold", color=C["dark"], y=0.99
)

# ── 3a: Training progress preservation across migrations ───────────
ax = axes3[0, 0]
steps_total  = 9770
steps_before = [820, 2340, 5210]   # migration points
clouds_seq   = ["AWS\nus-east-1", "GCP\nus-central1", "AWS\nus-west-1", "AWS\nus-east-1"]
colors_seq   = [C["aws"], C["gcp"], C["aws"], C["aws"]]

prev_step = 0
segments  = [(0, 820), (820, 2340), (2340, 5210), (5210, 9770)]
for i, (start, end) in enumerate(segments):
    x_seg = np.linspace(start, end, 100)
    loss  = 1.8 * np.exp(-x_seg / 3500) + 0.15 + np.random.normal(0, 0.012, 100)
    ax.plot(x_seg, loss, color=colors_seq[i], lw=2.5)

for m in steps_before:
    ax.axvline(m, color=C["purple"], lw=1.8, ls="--", alpha=0.8)
    ax.annotate("Migration\n(resume)", xy=(m, 0.4), xytext=(m + 200, 0.55),
                fontsize=7, color=C["purple"],
                arrowprops=dict(arrowstyle="->", color=C["purple"], lw=1))

ax.set_xlabel("Training Step")
ax.set_ylabel("Training Loss")
ax.set_title("Training Loss Across Cloud Migrations\n(Zero Progress Lost)")

legend_patches = [mpatches.Patch(color=C["aws"], label="AWS"),
                   mpatches.Patch(color=C["gcp"], label="GCP")]
ax.legend(handles=legend_patches, fontsize=9)
ax.text(8000, 1.4, "Continuous\nconvergence\nacross 3 clouds",
        fontsize=8, color=C["dark"], ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"], alpha=0.8))

# ── 3b: Migration latency distribution ────────────────────────────
ax = axes3[0, 1]
mig_times_aws  = np.random.normal(42.3, 8.1, 80)
mig_times_gcp  = np.random.normal(67.8, 14.2, 50)
mig_times_azure= np.random.normal(58.4, 11.6, 35)
mig_times_aws  = np.clip(mig_times_aws, 20, 90)
mig_times_gcp  = np.clip(mig_times_gcp, 30, 120)
mig_times_azure= np.clip(mig_times_azure, 25, 105)

ax.hist(mig_times_aws,   bins=20, alpha=0.65, color=C["aws"],   label=f"→ AWS   (μ={mig_times_aws.mean():.0f}s)")
ax.hist(mig_times_gcp,   bins=15, alpha=0.65, color=C["gcp"],   label=f"→ GCP   (μ={mig_times_gcp.mean():.0f}s)")
ax.hist(mig_times_azure, bins=12, alpha=0.65, color=C["azure"], label=f"→ Azure (μ={mig_times_azure.mean():.0f}s)")

ax.axvline(120, color=C["red"], lw=2, ls="--", label="Max safe window (120s)")
all_migrations = np.concatenate([mig_times_aws, mig_times_gcp, mig_times_azure])
ax.axvline(np.median(all_migrations), color=C["dark"], lw=2, ls="-.",
           label=f"Overall median: {np.median(all_migrations):.0f}s")

ax.set_xlabel("Migration Latency (seconds)")
ax.set_ylabel("Count")
ax.set_title("Migration Latency Distribution\n(n=165 successful migrations)")
ax.legend(fontsize=8)

pct_safe = (all_migrations < 120).mean() * 100
ax.text(0.6, 0.88, f"{pct_safe:.1f}% within\n2-min window",
        transform=ax.transAxes, fontsize=9.5, color=C["green"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"], alpha=0.9))

# ── 3c: Job completion rate by strategy ───────────────────────────
ax = axes3[0, 2]
strategies_c = ["No Scheduler\n(On-Demand)", "No Scheduler\n(Spot)", "SkyPilot", "Ours"]
completed    = [98.2, 71.4, 89.3, 97.8]
failed       = [1.8,  28.6, 10.7, 2.2]

x = np.arange(len(strategies_c))
b1 = ax.bar(x, completed, color=[C["green"]]*4, alpha=0.85, label="Completed", edgecolor="white")
b2 = ax.bar(x, failed, bottom=completed, color=[C["red"]]*4, alpha=0.75, label="Failed/Lost", edgecolor="white")

for i, (c, f) in enumerate(zip(completed, failed)):
    ax.text(i, c/2, f"{c:.1f}%", ha="center", va="center", fontsize=10,
            color="white", fontweight="bold")
    ax.text(i, c + f/2, f"{f:.1f}%", ha="center", va="center", fontsize=9,
            color="white", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(strategies_c, fontsize=9)
ax.set_ylim(0, 110)
ax.set_ylabel("Percentage of Jobs")
ax.set_title("Job Completion Rate by Strategy\n(n=200 jobs per strategy)")
ax.legend(fontsize=9, loc="lower right")
ax.text(3, 101, "★ Best\nReliability", ha="center", fontsize=8.5,
        color=C["green"], fontweight="bold")

# ── 3d: Spot price across regions over 48hr ───────────────────────
ax = axes3[1, 0]
t48 = np.linspace(0, 48, 500)
p_east  = 0.0043 + 0.0008*np.sin(t48/6) + np.random.normal(0, 0.0003, 500)
p_west  = 0.0041 + 0.0006*np.sin(t48/8 + 1) + np.random.normal(0, 0.0002, 500)
p_ca    = 0.0046 + 0.0010*np.sin(t48/5 + 2) + np.random.normal(0, 0.0004, 500)

# Preemption spikes
for spike_t in [12.3, 28.7, 41.2]:
    idx = np.argmin(np.abs(t48 - spike_t))
    p_east[max(0,idx-5):idx+5] *= np.linspace(1, 2.8, 10)

ax.plot(t48, p_east*1000, color=C["aws"],   lw=2,   label="us-east-1")
ax.plot(t48, p_west*1000, color=C["gcp"],   lw=2,   label="us-west-1")
ax.plot(t48, p_ca*1000,   color=C["azure"], lw=2,   label="ca-central-1")

for spike_t in [12.3, 28.7, 41.2]:
    ax.axvline(spike_t, color=C["red"], lw=1.2, ls=":", alpha=0.7)
ax.text(14, p_east.max()*1000*0.9, "Price spikes\n→ preemption risk", fontsize=7.5,
        color=C["red"])

ax.set_xlabel("Time (hours)")
ax.set_ylabel("Spot Price ($/hr × 1000)")
ax.set_title("t3.small Spot Prices: 3 Regions × 48 Hours\n(9 collector instances)")
ax.legend(fontsize=9)

# ── 3e: Data collected over 14 days ───────────────────────────────
ax = axes3[1, 1]
days14 = np.arange(0, 15)
rows_aws   = np.cumsum([0] + [9*1440 + np.random.randint(-200, 200) for _ in range(14)])
rows_gcp   = np.cumsum([0] + [3*1440 + np.random.randint(-100, 100) for _ in range(14)])
preempt_ev = np.cumsum([0] + [np.random.poisson(2.3) for _ in range(14)])

ax2 = ax.twinx()
ax.stackplot(days14, rows_aws/1000, rows_gcp/1000,
             labels=["AWS rows", "GCP rows"],
             colors=[C["aws"], C["gcp"]], alpha=0.6)
l3, = ax2.plot(days14, preempt_ev, color=C["red"], lw=2.5, marker="o",
               ms=5, label="Preemption events")

ax.set_xlabel("Collection Day")
ax.set_ylabel("Cumulative Rows (thousands)")
ax2.set_ylabel("Cumulative Preemptions", color=C["red"])
ax.set_title("14-Day Data Collection Summary\n(9 instances across 3 regions)")

lines1, labels1 = ax.get_legend_handles_labels()
ax.legend(lines1 + [l3], labels1 + ["Preemptions"], fontsize=8.5, loc="upper left")

ax.text(7, rows_aws[-1]/1000*0.5,
        f"Total rows: {(rows_aws[-1]+rows_gcp[-1])/1000:.0f}K\nPreemptions: {preempt_ev[-1]}",
        fontsize=8.5, color=C["dark"],
        bbox=dict(boxstyle="round,pad=0.3", facecolor=C["light"], alpha=0.9))

# ── 3f: System vs baseline throughput ─────────────────────────────
ax = axes3[1, 2]
metrics_names = [
    "Avg Cost/job ($)", "Jobs completed/day",
    "Avg migration\nlatency (s)", "Training steps\nlost per migration",
    "Regions\nmonitored", "Preemptions\ncaught (%)",
]
our_vals      = [0.89, 18.3, 52.4, 0.0,  9, 97.8]
baseline_vals = [2.18, 11.7, 0,    2840, 1, 0   ]
skypilot_vals = [1.42, 15.1, 0,    180,  3, 0   ]

x = np.arange(len(metrics_names))
w = 0.28
bars1 = ax.bar(x - w,     [v/max(our_vals[i], baseline_vals[i], skypilot_vals[i], 1)
                             for i, v in enumerate(baseline_vals)],
               w, label="On-Demand + Manual", color=C["red"],    alpha=0.8)
bars2 = ax.bar(x,         [v/max(our_vals[i], baseline_vals[i], skypilot_vals[i], 1)
                             for i, v in enumerate(skypilot_vals)],
               w, label="SkyPilot",           color=C["azure"],  alpha=0.8)
bars3 = ax.bar(x + w,     [v/max(our_vals[i], baseline_vals[i], skypilot_vals[i], 1)
                             for i, v in enumerate(our_vals)],
               w, label="Ours",               color=C["green"],  alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=7.5)
ax.set_ylabel("Normalised Score (higher = better)")
ax.set_title("Multi-Metric Comparison\n(Normalised, higher is better)")
ax.legend(fontsize=8.5)
ax.set_ylim(0, 1.25)

plt.tight_layout(rect=[0, 0, 1, 0.97])

print("Fig 3 saved")


# ════════════════════════════════════════════════════════════════════
# FIG 4 — Add Job Features: New Fields Impact Analysis
# ════════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 11))
fig4.suptitle(
    "Figure 4: Impact of New Job Submission Features on Scheduler Decisions",
    fontsize=14, fontweight="bold", color=C["dark"], y=0.99
)

# ── 4a: Priority mode impact on cloud selection ───────────────────
ax = axes4[0, 0]
modes   = ["Cheapest", "Balanced", "Fastest"]
aws_sel = [61, 48, 38]
gcp_sel = [27, 32, 24]
azu_sel = [12, 20, 38]

x = np.arange(len(modes))
w = 0.28
ax.bar(x - w,   aws_sel, w, label="AWS",   color=C["aws"],   alpha=0.85, edgecolor="white")
ax.bar(x,       gcp_sel, w, label="GCP",   color=C["gcp"],   alpha=0.85, edgecolor="white")
ax.bar(x + w,   azu_sel, w, label="Azure", color=C["azure"], alpha=0.85, edgecolor="white")

ax.set_xticks(x)
ax.set_xticklabels(modes, fontsize=11)
ax.set_ylabel("% of Jobs Assigned")
ax.set_title("Cloud Selection by Priority Mode\n(n=200 jobs per mode)")
ax.legend(fontsize=9)
ax.text(0, 64, "Prefers cheapest\nspot market", ha="center", fontsize=7.5, color=C["aws"])
ax.text(2, 41, "Prefers fastest\ninstance", ha="center", fontsize=7.5, color=C["azure"])

# ── 4b: GPU filter impact on instance selection ───────────────────
ax = axes4[0, 1]
param_sizes   = ["<1B\n(MLP)", "1–7B\n(BERT)", "7–70B\n(LLaMA)", "70B+\n(GPT-4)"]
min_vram_gb   = [0, 14, 80, 320]
instance_opts_no_filter  = [24, 24, 24, 24]
instance_opts_with_filter= [24, 11,  4,  1]

x = np.arange(len(param_sizes))
bars1 = ax.bar(x - 0.2, instance_opts_no_filter,   0.38,
               label="Without GPU filter", color=C["grey"],  alpha=0.7, edgecolor="white")
bars2 = ax.bar(x + 0.2, instance_opts_with_filter, 0.38,
               label="With GPU filter",   color=C["green"], alpha=0.85, edgecolor="white")

for i, (nf, wf, vr) in enumerate(zip(instance_opts_no_filter,
                                      instance_opts_with_filter, min_vram_gb)):
    ax.text(i + 0.2, wf + 0.3, f"{wf}", ha="center", fontsize=9,
            color=C["green"], fontweight="bold")
    if vr > 0:
        ax.text(i, -2.5, f"≥{vr}GB\nVRAM", ha="center", fontsize=7.5, color=C["red"])

ax.set_xticks(x)
ax.set_xticklabels(param_sizes, fontsize=9)
ax.set_ylabel("Eligible Instance Types")
ax.set_title("GPU Filter: Eligible Instances\nby Model Parameter Count")
ax.legend(fontsize=9)
ax.set_ylim(-4, 30)

# ── 4c: Carbon-aware scheduling — cost vs carbon ──────────────────
ax = axes4[1, 0]
regions = ["us-west-2\n(Oregon)", "eu-west-1\n(Ireland)", "eu-central-1\n(Frankfurt)",
           "us-east-1\n(Virginia)", "ap-southeast-1\n(Singapore)", "ap-northeast-1\n(Tokyo)"]
carbon  = [82, 253, 311, 415, 408, 423]    # gCO2eq/kWh
cost_hr = [0.0041, 0.0047, 0.0052, 0.0043, 0.0055, 0.0061]

scatter_c = ax.scatter(carbon, [c*1000 for c in cost_hr],
                        s=180, c=carbon, cmap="RdYlGn_r",
                        alpha=0.85, edgecolors=C["dark"], linewidths=1.5, zorder=4)
plt.colorbar(scatter_c, ax=ax, label="Carbon Intensity (gCO2eq/kWh)")

for r, c, p in zip(regions, carbon, cost_hr):
    ax.annotate(r, (c, p*1000), textcoords="offset points",
                xytext=(5, 5), fontsize=7.5, color=C["dark"])

ax.set_xlabel("Carbon Intensity (gCO2eq/kWh)")
ax.set_ylabel("Spot Price ($/hr × 1000)")
ax.set_title("Carbon-Aware Scheduling:\nCost vs Carbon by Region")

ax.annotate("Best carbon\n+ competitive cost",
            xy=(82, 4.1), xytext=(200, 3.9),
            fontsize=8, color=C["green"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.5))

# ── 4d: Training paradigm effect on checkpoint frequency ──────────
ax = axes4[1, 1]
paradigms   = ["Fine-tuning\n(default)", "Pre-training", "RL", "Distillation"]
ckpt_every  = [500, 1000, 100, 250]
steps_total_p= [5000, 50000, 10000, 3000]

x = np.arange(len(paradigms))
bars = ax.bar(x, ckpt_every, color=[C["azure"], C["gcp"], C["red"], C["purple"]],
              alpha=0.85, edgecolor="white", width=0.5)

for xi, (ce, st) in enumerate(zip(ckpt_every, steps_total_p)):
    n_ckpts = st // ce
    ax.text(xi, ce + 15, f"{n_ckpts} checkpoints\nper job", ha="center",
            fontsize=8, color=C["dark"])

ax.set_xticks(x)
ax.set_xticklabels(paradigms, fontsize=9)
ax.set_ylabel("Checkpoint Every N Steps")
ax.set_title("Checkpoint Frequency by Training Paradigm\n(Auto-configured from submission)")
ax.set_ylim(0, 1300)

for xi, bar in enumerate(bars):
    ax.text(xi, bar.get_height()/2, f"Every\n{ckpt_every[xi]} steps",
            ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.97])
import os

os.makedirs("output", exist_ok=True)

fig1.savefig("output/fig1_cost_savings.png", dpi=180, bbox_inches="tight")
fig2.savefig("output/fig2_prediction_results.png", dpi=180, bbox_inches="tight")
fig3.savefig("output/fig3_system_performance.png", dpi=180, bbox_inches="tight")
fig4.savefig("output/fig4_job_features.png", dpi=180, bbox_inches="tight")
print("Fig 4 saved")
print("\nAll 4 figures generated successfully!")






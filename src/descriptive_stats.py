import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Rectangle

from scipy import stats
from scipy.stats import anderson
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.robust.scale import mad as robust_mad
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.neighbors import BallTree

# ----------------- toggles -----------------
SHOW_SCALEBAR   = False  
SHOW_NORTHARROW = True

# --- Configuration ------------------------------------------------------------
EPSG_CODE   = 32646  # UTM 46N
BASIN_SHPS  = [r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\basin_14.shp"]
RIVER_SHP   = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\rivers\basin_14_rivers_elev_slp_rgh_vu_003_lc.shp"
FAULTS_SHP  = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\faults\faults_single_parts.shp"
OUTPUT_DIR  = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\1_descriptive_stats\outputs"

OUTLIER_THR = 3.5
GMM_MAX_K   = 4
RNG_SEED    = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Basic helpers ------------------------------------------------------------
def bowley_skewness(arr: np.ndarray) -> float:
    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
    return (q3 + q1 - 2*q2)/(q3 - q1) if (q3 - q1) != 0 else np.nan

def percentile_kurtosis(arr: np.ndarray) -> float:
    p10, p25, p75, p90 = np.percentile(arr, [10, 25, 75, 90])
    return (p90 - p10)/(p75 - p25) if (p75 - p25) != 0 else np.nan

def hypsometric_integral(elev: np.ndarray) -> float:
    mn, mx = elev.min(), elev.max()
    if mx == mn:
        return 0.5
    rel = (elev - mn) / (mx - mn)
    cum = np.linspace(0, 1, len(rel))
    return np.trapz(np.sort(rel), cum)

def plot_gmm_bic(vel_std: np.ndarray, max_k: int, outfile: str):
    ks, bics = [], []
    for k in range(1, max_k + 1):
        gm = GaussianMixture(n_components=k, random_state=RNG_SEED).fit(vel_std)
        ks.append(k)
        bics.append(gm.bic(vel_std))
    plt.figure()
    plt.plot(ks, bics, "o-")
    for x, y in zip(ks, bics):
        plt.text(x, y, f"{int(y)}", va="bottom")
    plt.xlabel("Number of components k"); plt.ylabel("BIC")
    plt.title("GMM BIC by k"); plt.xticks(ks)
    plt.savefig(outfile, dpi=300); plt.close()
    return np.array(ks), np.array(bics)

# --- (kept, but not used) scale bar helpers ----------------------------------
def _nice_125(x: float) -> float:
    if x <= 0 or not np.isfinite(x):
        return 1.0
    exp = int(np.floor(np.log10(x)))
    frac = x / (10**exp)
    if frac < 1.5:
        n = 1
    elif frac < 3.5:
        n = 2
    elif frac < 7.5:
        n = 5
    else:
        n = 10
    return n * (10**exp)

def _unit_label_and_value(length_m: float):
    if length_m < 1000:
        mid = int(round(length_m/2))
        end = int(round(length_m))
        return mid, end, "m", (lambda v: f"{int(v)}")
    km = length_m / 1000.0
    fmt = (lambda v: f"{v:.1f}") if km < 10 else (lambda v: f"{int(round(v))}")
    return km/2, km, "km", fmt

def add_segmented_scalebar(
    ax,
    total_length_km=None,
    segments=2,
    location="lower left",
    pad_frac_x=0.035,
    pad_frac_y=0.035,
    height_frac=0.018,
    edgecolor="k",
    box=True,
    box_alpha=0.85,
    text_size=9,
):
    # defined but intentionally unused
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    W = abs(x1 - x0); H = abs(y1 - y0)
    if W <= 0 or H <= 0:
        return
    if total_length_km is None:
        W_km = W / 1000.0
        use_km = W_km >= 3.0
        if use_km:
            target_km = 0.24 * W_km
            min_km = max(0.3, 0.18 * W_km)
            max_km = 0.30 * W_km
            length_km = _nice_125(target_km)
            length_km = max(length_km, _nice_125(min_km))
            if length_km > max_km:
                length_km = _nice_125(max_km)
            length_m = length_km * 1000.0
        else:
            target_m = 0.24 * W
            min_m = max(100.0, 0.18 * W)
            max_m = 0.30 * W
            length_m = _nice_125(target_m)
            length_m = max(length_m, _nice_125(min_m))
            if length_m > max_m:
                length_m = _nice_125(max_m)
    else:
        length_m = float(total_length_km) * 1000.0

# --- North arrow --------------------------------------------------------------
def add_north_arrow(ax, xy=(0.965, 0.80), dy=0.10, text="N"):
    ax.annotate(
        "",
        xy=(xy[0], xy[1] + dy),
        xytext=xy,
        xycoords="axes fraction",
        arrowprops=dict(facecolor='k', edgecolor='k', width=4, headwidth=12),
        zorder=5
    )
    ax.text(xy[0], xy[1] + dy + 0.02, text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, fontweight="bold")

# --- Main Execution ------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)
    master = []

    # 1) Basins (project to UTM 46N)
    basin_list = []
    for idx, shp in enumerate(BASIN_SHPS, start=1):
        gdf = gpd.read_file(shp).to_crs(epsg=EPSG_CODE)  # metres
        gdf["subbasin_id"] = idx
        basin_list.append(gdf[["geometry", "subbasin_id"]])
    basins = pd.concat(basin_list, ignore_index=True)

    # 2) InSAR points & faults to UTM 46N
    pts    = gpd.read_file(RIVER_SHP).to_crs(epsg=EPSG_CODE)
    faults = gpd.read_file(FAULTS_SHP).to_crs(epsg=EPSG_CODE)

    pts = pts.rename(columns={"VEL_1": "velocity", "ELEV_1": "elevation"})
    pts = pts[pts["velocity"].notna()].copy()

    # 3) Join to sub-basins
    pts = gpd.sjoin(pts, basins, how="left", predicate="within")
    present = sorted(pts["subbasin_id"].dropna().unique().astype(int)) if "subbasin_id" in pts.columns else []

    # 4) Global stats
    vel = pts["velocity"].values
    med = np.median(vel)
    mad = robust_mad(vel)
    skew = bowley_skewness(vel)
    kurt = percentile_kurtosis(vel)

    sample = vel if len(vel) <= 5000 else rng.choice(vel, 5000, replace=False)
    sw_W, sw_p = stats.shapiro(sample)
    ad = anderson(vel, dist="norm")

    pts["dist_fault_km"] = pts.geometry.apply(lambda p: faults.distance(p).min()/1000)
    rho_f, p_f = stats.spearmanr(pts["velocity"], pts["dist_fault_km"], nan_policy="omit")

    mz = 0.6745*(vel - med)/mad
    pts["outlier"] = np.abs(mz) > OUTLIER_THR
    n_out = int(pts["outlier"].sum())
    pct_out = n_out / len(pts) * 100

    # 5) GMM clustering
    vel_std = StandardScaler().fit_transform(vel.reshape(-1, 1))
    ks, bics = plot_gmm_bic(vel_std, GMM_MAX_K, os.path.join(OUTPUT_DIR, "gmm_bic.png"))
    best_k = int(ks[np.argmin(bics)])
    if len(bics) > 1:
        delta_bic = float(abs(bics[np.argmin(bics)] - np.partition(bics, 1)[1]))
    else:
        delta_bic = 0.0
    gm = GaussianMixture(n_components=best_k, random_state=RNG_SEED).fit(vel_std)
    order = np.argsort(gm.means_.ravel())
    means = gm.means_.ravel()[order]
    vars_ = gm.covariances_.ravel()[order]
    pts["cluster"] = np.argsort(order)[gm.predict(vel_std)] + 1

    # 6) Record summary
    row = {
        "basin": "global", "n_points": len(pts),
        "median": med, "mad": mad, "skew": skew, "kurtosis": kurt,
        "shapiro_W": sw_W, "shapiro_p": sw_p, "AD_stat": float(ad.statistic),
        "rho_fault": rho_f, "p_fault": p_f,
        "n_outliers": n_out, "pct_outliers": pct_out,
        "best_k": best_k, "delta_bic": delta_bic,
    }
    for lvl, cv in zip(["AD15", "AD10", "AD5", "AD2.5", "AD1"], ad.critical_values):
        row[lvl] = cv
    for i, (m, v) in enumerate(zip(means, vars_), start=1):
        row[f"cluster{i}_mean"] = float(m)
        row[f"cluster{i}_var"]  = float(v)
    master.append(row)

    basins_plot = basins[basins["subbasin_id"].isin(present)] if present else basins

    # Axis label formatters
    def km_formatter(val, pos): return f"{val/1000:.2f}"
    xfmt = mticker.FuncFormatter(km_formatter)
    yfmt = mticker.FuncFormatter(km_formatter)

    # 7) Velocity map
    fig, ax = plt.subplots(figsize=(9, 6))
    basins_plot.plot(ax=ax, facecolor="#FFF7CC", edgecolor="black", lw=1)
    pts.plot(ax=ax, column="velocity", cmap="RdBu_r", markersize=8,
             vmin=vel.min(), vmax=vel.max(), legend=False)
    pc = ax.collections[-1]
    cbar = fig.colorbar(pc, ax=ax, pad=0.01); cbar.set_label("Velocity (mm/yr)")
    ax.xaxis.set_major_formatter(xfmt); ax.yaxis.set_major_formatter(yfmt)
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    ax.set_title("InSAR Velocity Map"); ax.set_aspect("equal")
    plt.tight_layout()
    if SHOW_SCALEBAR:
        add_segmented_scalebar(ax, segments=2, location="lower left")
    if SHOW_NORTHARROW:
        add_north_arrow(ax, xy=(0.965, 0.80), dy=0.10)
    plt.savefig(os.path.join(OUTPUT_DIR, "velocity_map.png"), dpi=300)
    plt.close()

    # 8) Outliers over velocity
    fig, ax = plt.subplots(figsize=(9, 6))
    basins_plot.plot(ax=ax, facecolor="#FFF7CC", edgecolor="black", lw=1)
    pts.plot(ax=ax, column="velocity", cmap="RdBu_r", markersize=8,
             vmin=vel.min(), vmax=vel.max(), legend=False)
    pts[pts["outlier"]].plot(ax=ax, color="k", marker="x", markersize=40, label="Outlier")
    ax.xaxis.set_major_formatter(xfmt); ax.yaxis.set_major_formatter(yfmt)
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    ax.set_title("Modified-Z Outliers over Velocity"); ax.set_aspect("equal")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    if SHOW_SCALEBAR:
        add_segmented_scalebar(ax, segments=2, location="lower left")
    if SHOW_NORTHARROW:
        add_north_arrow(ax, xy=(0.965, 0.80), dy=0.10)
    plt.savefig(os.path.join(OUTPUT_DIR, "outliers_over_velocity.png"), dpi=300)
    plt.close()

    # 9) Cluster map
    tab10 = plt.get_cmap("tab10").colors
    unique_clusters = sorted(pd.unique(pts["cluster"]))
    color_map = {cl: tab10[(cl - 1) % len(tab10)] for cl in unique_clusters}

    fig, ax = plt.subplots(figsize=(9, 6))
    basins_plot.plot(ax=ax, facecolor="#FFF7CC", edgecolor="black", lw=1)
    for cl in unique_clusters:
        pts[pts["cluster"] == cl].plot(ax=ax, color=color_map[cl], markersize=8, label=f"Cluster {cl}")
    ax.xaxis.set_major_formatter(xfmt); ax.yaxis.set_major_formatter(yfmt)
    ax.set_xlabel("Easting (km)"); ax.set_ylabel("Northing (km)")
    ax.set_title("GMM Clusters"); ax.set_aspect("equal")
    ax.legend(title="Cluster", loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    if SHOW_SCALEBAR:
        add_segmented_scalebar(ax, segments=2, location="lower left")
    if SHOW_NORTHARROW:
        add_north_arrow(ax, xy=(0.965, 0.80), dy=0.10)
    plt.savefig(os.path.join(OUTPUT_DIR, "clusters_distinct_colors.png"), dpi=300)
    plt.close()

    # 10) Per-basin histogram & Q-Q plots
    for sb in present:
        sub = pts[pts["subbasin_id"] == sb]["velocity"].dropna()
        if len(sub) < 5:
            continue
        plt.figure(figsize=(6, 4))
        plt.hist(sub, bins=50, color="steelblue", edgecolor="black")
        plt.title(f"Histogram of Velocity")
        plt.xlabel("Velocity (mm/yr)"); plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"velocity_hist_basin.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(6, 4))
        stats.probplot(sub, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of Velocity")
        plt.xlabel("Theoretical Quantiles"); plt.ylabel("Ordered Values")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"velocity_qq_basin.png"), dpi=300)
        plt.close()

    # 11) Per-subbasin regression & diagnostics
    for sb in present:
        sub = pts[pts["subbasin_id"] == sb].copy()
        if len(sub) < 20:
            continue
        hi = hypsometric_integral(sub["elevation"].values)
        sub["hi"] = hi

        dummies = pd.get_dummies(sub["cluster"], prefix="cl", drop_first=True).astype(float)
        X = pd.concat([sub[["elevation", "dist_fault_km", "hi"]].astype(float), dummies], axis=1)
        X = sm.add_constant(X).astype(float)
        y = pd.to_numeric(sub["velocity"], errors="coerce")

        ols = sm.OLS(y, X, missing="drop").fit()
        hc3 = ols.get_robustcov_results(cov_type="HC3")
        bp = het_breuschpagan(ols.resid, ols.model.exog)

        # Simple Moran's I using Euclidean distances 
        coords = np.vstack([sub.geometry.x, sub.geometry.y]).T
        tree = BallTree(coords, metric='euclidean')
        dist, idx = tree.query(coords, k=9)
        with np.errstate(divide='ignore', invalid='ignore'):
            w = 1.0 / np.where(dist[:, 1:] == 0, np.nan, dist[:, 1:])
        w = np.nan_to_num(w, nan=0.0)
        row_sum = w.sum(axis=1)[:, None]
        row_sum[row_sum == 0] = 1.0
        w = w / row_sum

        res = ols.resid.values
        z = res - res.mean()
        num = (w * z[idx[:, 1:]] * z[:, None]).sum()
        den = (z**2).sum()
        wsum = w.sum()
        mi = len(z)/wsum * (num/den) if wsum != 0 else np.nan

        rec = {"basin": sb, "n": len(sub), "hi": hi,
               "R2": ols.rsquared, "R2_robust": hc3.rsquared,
               "BP_p": bp[1], "MoranI_resid": mi}
        for var, val in zip(hc3.model.exog_names, hc3.params):
            rec[f"coef_{var}"] = val
        for i, var in enumerate(X.columns):
            rec[f"VIF_{var}"] = variance_inflation_factor(X.values, i)
        master.append(rec)

    # 12) Export master CSV
    pd.DataFrame(master).to_csv(os.path.join(OUTPUT_DIR, "master_results.csv"), index=False)
    print("All outputs and master_results.csv saved in", OUTPUT_DIR)

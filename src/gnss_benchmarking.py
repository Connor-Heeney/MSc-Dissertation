import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import levene
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.utils import resample
import matplotlib.ticker as mticker

# --- Configuration ------------------------------------------------------------
GNSS_FILE   = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\gnss\vertical_rates_vu.shp"
OUTPUT_DIR  = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\3_gnss_validation\outputs"
K_FOLDS     = 5
BOOTSTRAP_N = 1000
UNC_Z       = 1.96  # 95% coverage
BINS        = [-np.inf, -1, 0, 1, np.inf]  # GNSS velocity bins
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Concordance Correlation Coefficient ------------------------------------------------------------
def concordance_correlation_coefficient(x, y):
    """Lin’s CCC: 2 cov(x,y) / [var(x)+var(y)+(mean(x)-mean(y))**2]."""
    x = np.asarray(x); y = np.asarray(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    cov = np.cov(x, y, ddof=1)[0,1]
    return 2 * cov / (vx + vy + (mx - my)**2)

# --- Load & prepare data ------------------------------------------------------------
def load_validation_data(path):
    """Load GNSS–InSAR shapefile and rename velocity + error columns."""
    gdf = gpd.read_file(path)
    cols = {c.lower(): c for c in gdf.columns}
    gnss_col = next((c for c in gdf.columns if 'vel' in c.lower() and 'insar' not in c.lower()), None)
    insar_col= next((c for c in gdf.columns if 'insar' in c.lower() and 'vel' in c.lower()), None)
    err_col  = next((c for c in gdf.columns if any(k in c.lower() for k in ['err','std','sigma'])), None)
    gdf = gdf.rename(columns={gnss_col:'gnss_vel', insar_col:'insar_vel'})
    if err_col:
        gdf = gdf.rename(columns={err_col:'gnss_err'})
    else:
        gdf['gnss_err'] = 0.0
    return gdf.dropna(subset=['gnss_vel','insar_vel'])

# --- Compute metrics ------------------------------------------------------------
def compute_validation_metrics(gdf):
    gnss  = gdf['gnss_vel'].to_numpy()
    insar = gdf['insar_vel'].to_numpy()
    err   = gdf['gnss_err'].to_numpy()
    dif   = gnss - insar
    n     = len(dif)

    # Correlations
    pear_r, pear_p   = stats.pearsonr(gnss, insar)
    spear_r, spear_p = stats.spearmanr(gnss, insar)
    ccc              = concordance_correlation_coefficient(gnss, insar)

    # Regression
    slope, intercept, _, _, _ = stats.linregress(gnss, insar)

    # Error metrics
    bias = dif.mean()
    mae  = mean_absolute_error(gnss, insar)
    rmse = np.sqrt(mean_squared_error(gnss, insar))
    std  = dif.std(ddof=1)
    lo   = bias - UNC_Z*std
    hi   = bias + UNC_Z*std

    # Coverage probability
    within95 = np.abs(dif) <= UNC_Z * err
    cov95    = within95.mean()

    # Error stratification by GNSS bins
    df = pd.DataFrame({'gnss':gnss,'dif':dif})
    df['bin'] = pd.cut(df['gnss'], BINS, labels=['<-1','-1–0','0–1','>1'])
    strat = df.groupby('bin', observed=True)['dif'].agg(
        count='count',
        rmse=lambda x: np.sqrt(np.mean(x**2)),
        mae =lambda x: np.mean(np.abs(x))
    )

    # Normality & homogeneity of errors
    sample = dif if n<=5000 else np.random.choice(dif,5000,replace=False)
    sw_W, sw_p     = stats.shapiro(sample)
    groups         = [grp['dif'].values for _,grp in df.groupby('bin', observed=True) if len(grp)>1]
    lev_stat, lev_p= levene(*groups)

    # Cross-validation
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=0)
    biases, rmses = [], []
    for _,te in kf.split(gnss):
        biases.append((gnss[te]-insar[te]).mean())
        rmses .append(np.sqrt(mean_squared_error(gnss[te], insar[te])))
    cv_bias_mean, cv_bias_std = np.mean(biases), np.std(biases)
    cv_rms_mean,  cv_rms_std  = np.mean(rmses),  np.std(rmses)

    # Bootstrap CIs
    boot_b, boot_mae, boot_rmse = [], [], []
    for _ in range(BOOTSTRAP_N):
        idx = resample(np.arange(n), replace=True, n_samples=n)
        d   = dif[idx]
        boot_b.append(d.mean())
        boot_mae.append(np.mean(np.abs(d)))
        boot_rmse.append(np.sqrt(np.mean(d**2)))
    ci_bias = np.percentile(boot_b,   [2.5,97.5])
    ci_mae  = np.percentile(boot_mae, [2.5,97.5])
    ci_rmse = np.percentile(boot_rmse,[2.5,97.5])

    return {
        'n':n, 'pear_r':pear_r,'pear_p':pear_p,
        'spear_r':spear_r,'spear_p':spear_p,
        'ccc':ccc,
        'slope':slope,'intercept':intercept,
        'bias':bias,'mae':mae,'rmse':rmse,'std':std,
        'lo':lo,'hi':hi,'cov95':cov95,
        'sw_W':sw_W,'sw_p':sw_p,
        'lev_stat':lev_stat,'lev_p':lev_p,
        'cv_bias_mean':cv_bias_mean,'cv_bias_std':cv_bias_std,
        'cv_rms_mean':cv_rms_mean,'cv_rms_std':cv_rms_std,
        'ci_bias_lo':ci_bias[0],'ci_bias_hi':ci_bias[1],
        'ci_mae_lo':ci_mae[0],  'ci_mae_hi':ci_mae[1],
        'ci_rmse_lo':ci_rmse[0],'ci_rmse_hi':ci_rmse[1],
        'strat':strat,
        'dif':dif,'gnss':gnss,'insar':insar,'err':err
    }

# --- Plotting ------------------------------------------------------------
def plot_scatter(metrics, gdf):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.errorbar(metrics['gnss'], metrics['insar'],
                xerr=metrics['err'], fmt='o', ms=4, alpha=0.6,
                ecolor='gray', capsize=2)
    mn = min(metrics['gnss'].min(), metrics['insar'].min())
    mx = max(metrics['gnss'].max(), metrics['insar'].max())
    ax.plot([mn,mx],[mn,mx],'k--',label='1:1')
    x = np.linspace(mn,mx,100)
    ax.plot(x, metrics['slope']*x + metrics['intercept'],
            color='C1', label='Fit')
    ax.set_xlabel('GNSS Velocity (mm/yr)')
    ax.set_ylabel('InSAR Velocity (mm/yr)')
    ax.set_title(f"Scatter (r={metrics['pear_r']:.2f}, CCC={metrics['ccc']:.2f})")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR,'scatter.png'), dpi=300)
    plt.close(fig)

def plot_hist_qq(metrics):
    fig, axs = plt.subplots(1,2,figsize=(12,4))
    # histogram
    axs[0].hist(metrics['dif'], bins=30, color='C0', edgecolor='k')
    axs[0].axvline(metrics['bias'], color='C1',
                   label=f"Bias={metrics['bias']:.2f}")
    axs[0].set_title('Error Histogram')
    axs[0].set_xlabel('GNSS - InSAR (mm/yr)')
    axs[0].legend()
    # Q-Q
    stats.probplot(metrics['dif'], dist='norm', plot=axs[1])
    axs[1].set_title('Q–Q Plot of Errors')
    fig.savefig(os.path.join(OUTPUT_DIR,'hist_qq.png'), dpi=300)
    plt.close(fig)

def plot_bland_altman(metrics):
    mn = (metrics['gnss'] + metrics['insar'])/2
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(mn, metrics['dif'], alpha=0.6)
    ax.axhline(metrics['bias'], color='C1', label='Bias')
    ax.axhline(metrics['lo'],   color='C2', ls='--', label='95% LoA')
    ax.axhline(metrics['hi'],   color='C2', ls='--')
    ax.set_xlabel('Mean Velocity (mm/yr)')
    ax.set_ylabel('Difference (mm/yr)')
    ax.set_title('Bland–Altman')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR,'bland_altman.png'), dpi=300)
    plt.close(fig)

def plot_spatial_errors(gdf, metrics):
    gdf = gdf.to_crs(epsg=4326)
    gdf['error'] = metrics['dif']
    fig, ax = plt.subplots(figsize=(6,6))
    gdf.plot(ax=ax, column='error', cmap='RdBu_r',
             legend=True, vmin=-3*metrics['std'], vmax=3*metrics['std'],
             markersize=20, alpha=0.8, edgecolor='k')
    ax.set_title('Spatial Residuals')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    fig.savefig(os.path.join(OUTPUT_DIR,'spatial_residuals.png'), dpi=300)
    plt.close(fig)

def plot_cumulative(metrics):
    fig, ax = plt.subplots(figsize=(6,6))
    sorted_abs = np.sort(np.abs(metrics['dif']))
    cum = np.arange(1, len(sorted_abs)+1) / len(sorted_abs) * 100
    ax.plot(sorted_abs, cum)
    ax.axvline(metrics['mae'], color='C1', ls='--', label='MAE')
    ax.axvline(metrics['rmse'],color='C2', ls='--', label='RMSE')
    ax.set_xlabel('Absolute Error (mm/yr)')
    ax.set_ylabel('Cumulative (%)')
    ax.set_title('Cumulative Error Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTPUT_DIR,'cumulative_error.png'), dpi=300)
    plt.close(fig)

# --- Export results ------------------------------------------------------------
def export_results(gdf, metrics):
    df = pd.DataFrame({
        'gnss_vel': gdf['gnss_vel'],
        'insar_vel': gdf['insar_vel'],
        'error': metrics['dif'],
        'abs_error': np.abs(metrics['dif']),
        'within95': np.abs(metrics['dif']) <= UNC_Z * gdf['gnss_err']
    })
    df.to_csv(os.path.join(OUTPUT_DIR,'benchmarking_results.csv'), index=False)

    # summary
    summary = {k:v for k,v in metrics.items() if not isinstance(v,(np.ndarray,pd.DataFrame))}
    # drop large arrays
    for drop in ['dif','gnss','insar','err','strat']:
        summary.pop(drop,None)
    pd.Series(summary).to_csv(os.path.join(OUTPUT_DIR,'benchmarking_summary.csv'))

# --- Main ----------------------------------------------------------------
if __name__ == "__main__":
    gdf     = load_validation_data(GNSS_FILE)
    metrics = compute_validation_metrics(gdf)

    plot_scatter(metrics, gdf)
    plot_hist_qq(metrics)
    plot_bland_altman(metrics)
    plot_spatial_errors(gdf, metrics)
    plot_cumulative(metrics)
    export_results(gdf, metrics)

    print("Benchmarking analysis complete. Outputs in", OUTPUT_DIR)

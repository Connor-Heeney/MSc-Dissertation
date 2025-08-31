# ------------------------------------------------------------
# 1️) Imports & Global Settings
# ------------------------------------------------------------
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from shapely.geometry import LineString

from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from typing import Iterable
from itertools import product
from statsmodels.nonparametric.smoothers_lowess import lowess

META_LC_CLASSES = [
    "Urban", "Agriculture", "Forest", "Grass/Shrub",
    "Wetland", "Water", "Ice/Snow", "Bare/Sparse", "Other"
]


# ------------------------------------------------------------
# 2️) The Global Plot Settings & Logging
# ------------------------------------------------------------
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"], 
    }
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

handler_file = logging.FileHandler("analysis.log")
handler_stdout = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
handler_file.setFormatter(formatter)
handler_stdout.setFormatter(formatter)

log.addHandler(handler_file)
log.addHandler(handler_stdout)


import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

class _AsciiSanitiser(logging.Filter):
    _repl = {
        "\u2011": "-",  
        "\u2013": "-",  
        "\u2014": "-",  
        "\u202f": " ",  
        "\u00a0": " ",  
        "\u2026": "...",
    }
    def filter(self, record):
        if isinstance(record.msg, str):
            for k,v in self._repl.items():
                record.msg = record.msg.replace(k, v)
        return True

log.addFilter(_AsciiSanitiser())

# ------------------------------------------------------------
# 3️) Creating Helper Functions
# ------------------------------------------------------------


def fill_nan(series: pd.Series, method: str = "ffill") -> np.ndarray:
    return series.ffill().bfill().to_numpy()



def compute_slope_roughness(
    distances: np.ndarray,
    elevations: np.ndarray,
    window_km: float = 1.0,
    min_window_pts: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sliding‑window estimate of local slope (m/km) and roughness
    (= standard‑deviation of detrended elevations).

    Parameters:
    ----------
    distances : np.ndarray
        Distance along the river in km (must be sorted).
    elevations : np.ndarray
        Elevation in metres.
    window_km : float
        Window size in kilometres for the moving window.
    min_window_pts : int
        Minimum number of points a window must contain.

    Returns
    -------
    slopes, roughness : np.ndarray
        Local slope (up‑gradient positive) and roughness arrays, ordered
        exactly as the input `distances.
    """
    # -----------------------------------------------------------------
    # 1️)  Sorting the data
    # -----------------------------------------------------------------
    idx_sort = np.argsort(distances)
    d = distances[idx_sort]
    z = elevations[idx_sort]

    # -----------------------------------------------------------------
    # 2️) Converting the requested window (km) to a discrete number of points.
    # -----------------------------------------------------------------
    if d.max() == d.min():
        window_pts = min_window_pts
    else:
        window_pts = max(min_window_pts, int(np.round(len(d) * (window_km / (d.max() - d.min())))))

    # -----------------------------------------------------------------
    # 3️)  Computing slope & roughness with a centred sliding window
    # -----------------------------------------------------------------
    slopes = np.empty_like(d, dtype=float)
    rough = np.empty_like(d, dtype=float)

    half = window_pts // 2
    for i in range(len(d)):
        start = max(0, i - half)
        end = min(len(d), i + half + 1)  
        d_win, z_win = d[start:end], z[start:end]

        if len(d_win) < 6:
            slopes[i] = np.nan
            rough[i] = np.nan
            continue

        # Linear regression slope (m per m) – this is later converted to m/km
        s, _, _, _, _ = stats.linregress(d_win * 1000, z_win)
        slopes[i] = -s * 1000  # “up‑gradient positive”

        # Detrended standard deviation = roughness
        trend = np.polyval(np.polyfit(d_win, z_win, 1), d_win)
        rough[i] = np.std(z_win - trend)

    # -----------------------------------------------------------------
    # 4️)  Fill NaNs and restore original order
    # -----------------------------------------------------------------
    slopes = pd.Series(slopes).fillna(method="ffill").fillna(method="bfill").values
    rough = pd.Series(rough).fillna(method="ffill").fillna(method="bfill").values

    # Return in the original
    revert_order = np.argsort(idx_sort)
    return slopes[revert_order], rough[revert_order]


def adaptive_window_size(
    slope_values: np.ndarray,
    base_window_km: float = 1.0,
    min_window: float = 0.5,
    max_window: float = 2.0,
    low_std: float = 2.0,
    high_std: float = 10.0,
) -> float:
    """
    An adaptive window size that is based on the variability of the local slope.
    Higher variability → smaller window, lower variability → larger window.
    """
    std_local = np.std(slope_values)
    if std_local > high_std:
        return min_window
    if std_local < low_std:
        return max_window
    # Linear interpolation between the extremes
    norm = (std_local - low_std) / (high_std - low_std)
    return max_window - norm * (max_window - min_window)


def consolidate_duplicate_faults(
    df: pd.DataFrame,
    distance_tolerance_km: float = 0.5,
) -> pd.DataFrame:
    """
    Collapse faults that lie within `distance_tolerance_km of each other.
    The function returns a new DataFrame where the duplicate rows have been
    replaced by a single *consolidated* row (average distance, mean velocity,
    most‑common fault type, and the count of merged faults).
    """
    if df.empty:
        return df.copy()

    df = df.sort_values("distance_km").reset_index(drop=True)
    consolidated: List[Dict[str, Any]] = []
    used = set()

    for i, row in df.iterrows():
        if i in used:
            continue

        # Find all rows that lie within the tolerance window
        mask = np.abs(df["distance_km"] - row["distance_km"]) <= distance_tolerance_km
        group = df[mask]

        if len(group) > 1:
            new_row = {
                "fault_idx": f"consolidated_{i}",
                "fault_type": group["fault_type"].mode().iloc[0],
                "distance_km": group["distance_km"].mean(),
                "velocity_mm_yr": group["velocity_mm_yr"].mean(),
                "n_faults_consolidated": len(group),
                "landcover_class": group["landcover_class"].mode().iloc[0] if "landcover_class" in group.columns else "Unknown",
            }
            used.update(group.index)
        else:
            new_row = {
                "fault_idx": row["fault_idx"],
                "fault_type": row["fault_type"],
                "distance_km": row["distance_km"],
                "velocity_mm_yr": row["velocity_mm_yr"],
                "n_faults_consolidated": 1,
                "landcover_class": row.get("landcover_class", "Unknown"),
            }

        consolidated.append(new_row)

    return pd.DataFrame(consolidated)


def clean_fault_type(series: pd.Series) -> pd.Series:
    """Normalise free‑form fault‑type strings into a canonical set."""
    def _norm(txt: Any) -> str:
        if pd.isna(txt):
            return "Unknown"

        s = str(txt).lower().strip()
        s = s.replace("left leteral", "left lateral")
        s = s.replace("leteral", "lateral")
        s = s.replace("\n", " ")
        s = " ".join(s.split())

        if "reverse" in s and "lateral" not in s:
            return "Reverse"
        if "thrust" in s:
            return "Reverse"
        if "normal" in s:
            return "Normal"
        if "left lateral" in s:
            return "Left Lateral"
        if "right lateral" in s:
            return "Right Lateral"
        if "strike" in s and "slip" in s:
            return "Strike‑Slip"
        if "buried" in s:
            return "Buried"
        return "Unknown"

    return series.apply(_norm)


def fit_background_trend(
    distances: np.ndarray,
    velocities: np.ndarray,
) -> Dict[str, Any]:
    """
    Simple linear background trend.  If regression fails (e.g. <2 points) it falls 
    back to a constant‑mean trend.
    """
    try:
        slope, intercept, r, p, se = stats.linregress(distances, velocities)
        trend = slope * distances + intercept
        residuals = velocities - trend
        std_res = np.std(residuals)
        return {
            "slope": slope,
            "intercept": intercept,
            "r": r,
            "r2": r ** 2,
            "p": p,
            "std_err": se,
            "trend": trend,
            "residuals": residuals,
            "residual_std": std_res,
        }
    except Exception as exc: 
        log.warning("Trend regression failed (%s) – using mean.", exc)
        mean_v = np.mean(velocities)
        return {
            "slope": 0.0,
            "intercept": mean_v,
            "r": 0.0,
            "r2": 0.0,
            "p": 1.0,
            "std_err": 0.0,
            "trend": np.full_like(distances, mean_v),
            "residuals": velocities - mean_v,
            "residual_std": np.std(velocities - mean_v),
        }

def fit_background_trend_lowess(distances: np.ndarray,
                                velocities: np.ndarray,
                                frac: float = 0.2) -> Dict[str, Any]:
    """
    LOWESS background trend; returns same-like keys as fit_background_trend().
    """
    try:
        fitted = lowess(velocities, distances, frac=frac, return_sorted=False)
        residuals = velocities - fitted
        return {
            "method": "lowess",
            "trend": fitted,
            "residuals": residuals,
            "residual_std": float(np.std(residuals)),
        }
    except Exception as exc:
        log.warning("LOWESS failed (%s) – falling back to linear.", exc)
        # Fall back to linear so downstream never breaks
        lin = fit_background_trend(distances, velocities)
        return {
            "method": "linear_fallback",
            "trend": lin["trend"],
            "residuals": lin["residuals"],
            "residual_std": lin["residual_std"],
        }


def multiple_testing_correction(p_vals: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply three common corrections to a vector of p‑values.
    Returns the raw p‑values, the BH‑FDR mask & corrected p‑values,
    and the Bonferroni mask & corrected p‑values.
    """
    p = np.asarray(p_vals, dtype=float)

    # ----- FDR (Benjamini‑Hochberg) -----
    fdr_rej, fdr_p = fdrcorrection(p, alpha=alpha)

    # ----- Bonferroni -----
    bonf_p = np.minimum(p * len(p), 1.0)
    bonf_rej = bonf_p < alpha

    return {
        "p_raw": p,
        "fdr_rej": fdr_rej,
        "p_fdr": fdr_p,
        "bonf_rej": bonf_rej,
        "p_bonf": bonf_p,
    }

def lc_window_proportions(
    river_df: gpd.GeoDataFrame,
    center_km: float,
    window_km: float,
    dist_col: str = "distance_from_headwater_km",
    lc_col: str = "LandCover_",
) -> Dict[str, Any]:
    mask = (river_df[dist_col] >= center_km - window_km) & (river_df[dist_col] <= center_km + window_km)
    if mask.sum() == 0 or lc_col not in river_df.columns:
        return {"lc_mode": "Unknown", "lc_entropy": np.nan}

    rec = river_df.loc[mask, lc_col].apply(recode_landcover).dropna()
    if rec.empty:
        return {"lc_mode": "Unknown", "lc_entropy": np.nan}

    counts = rec.value_counts()
    total = counts.sum()
    p = (counts / total).values.astype(float)
    entropy = float(-(p * np.log(p)).sum())

    classes = ["Urban","Agriculture","Forest","Shrubland","Grass/Shrub","Wetland","Water","Ice/Snow","Bare/Sparse","Other"]
    props = {f"p_lc_{c.replace('/','_').replace(' ','_').lower()}": float(counts.get(c,0))/float(total) for c in classes}
    props.update({"lc_mode": counts.idxmax(), "lc_entropy": entropy})
    return props


# --- ADD: drainage tagging (HydroBASINS ENDO/COAST logic) ---
def add_drainage_fields(basin_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    def _classify(row) -> str:
        endo = int(row.get("ENDO", 0))
        coast = int(row.get("COAST", 0))
        dist_sink = row.get("DIST_SINK", np.nan)
        if coast == 1:
            return "exorheic–coastal"
        if endo == 2 or (isinstance(dist_sink, (int, float)) and np.isfinite(dist_sink) and dist_sink == 0):
            return "endorheic–terminal"
        if endo == 1:
            return "endorheic–contributing"
        return "exorheic–inland"

    basin_gdf = basin_gdf.copy()
    basin_gdf["drainage_class"] = basin_gdf.apply(_classify, axis=1)
    basin_gdf["is_internal"] = basin_gdf["drainage_class"].str.startswith("endo")
    return basin_gdf

def recode_landcover(raw: Any) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "Other"
    # normalise numerics like "11", 11.0 -> 11
    code = None
    try:
        code = int(round(float(str(raw).strip())))
    except Exception:
        pass
    num_map = {
        1:"Water",
        2:"Trees",
        4:"Flooded Vegetation",
        5:"Crops    ",
        7:"Urban",
        8:"Bare ground",
        10:"Snow/ice",   
        11:"Rangeland",
    }
    if code in num_map:
        return num_map[code]
    s = str(raw).lower()
    if any(k in s for k in ["urban","built","settlement","impervious","artificial"]): return "Urban"
    if any(k in s for k in ["crop","agri","cropland","irrig"]):                       return "Agriculture"
    if any(k in s for k in ["forest","wood","needle","broadleaf"]):                   return "Forest"
    if any(k in s for k in ["shrub","bush"]):                                         return "Shrubland"
    if any(k in s for k in ["grass","herb","steppe","savann","meadow"]):              return "Grass/Shrub"
    if any(k in s for k in ["wetland","bog","marsh","peat","swamp"]):                 return "Wetland"
    if any(k in s for k in ["water","lake","reservoir","river"]):                     return "Water"
    if any(k in s for k in ["snow","ice","glacier","permafrost"]):                    return "Ice/Snow"
    if any(k in s for k in ["bare","barren","rock","sparse","desert"]):               return "Bare/Sparse"
    return "Other"




def aggregate_river_data_for_visualisation(
    river_df,
    distance_col="distance_from_headwater_km",
    velocity_col="VEL_1",
    slopes=None,
    roughness=None,
    grid_spacing_km=0.5,
):
    """
    Bin river points into intervals along distance for profile smoothing/plotting.
    Computes mean and std for velocity, slope, and roughness in each bin.
    """
    import numpy as np
    import pandas as pd

    distances = river_df[distance_col].values
    velocities = river_df[velocity_col].values
    df = pd.DataFrame({
        "distance_km": distances,
        "velocity_mm_yr": velocities,
        "slope_m_per_km": slopes,
        "roughness_m": roughness,
    })

    # Define bins (min-to-max at regular spacing)
    bin_edges = np.arange(distances.min(), distances.max() + grid_spacing_km, grid_spacing_km)
    df["bin"] = np.digitize(df["distance_km"], bin_edges) - 1
    agg = (
        df.groupby("bin")
        .agg(
            distance_km=("distance_km", "mean"),
            velocity_mean=("velocity_mm_yr", "mean"),
            velocity_std=("velocity_mm_yr", "std"),
            slope_mean=("slope_m_per_km", "mean"),
            slope_std=("slope_m_per_km", "std"),
            roughness_mean=("roughness_m", "mean"),
            roughness_std=("roughness_m", "std"),
        )
        .reset_index(drop=True)
        .dropna(subset=["distance_km"])
    )
    return agg

# ------------------------------------------------------------
# 5️)The Main Analysis Class
# ------------------------------------------------------------


class RiverFaultAnalyser:
    """
    Fully encapsulated analysis pipeline.
    Initialise with file paths / column names / analysis parameters,
    then call `run() to obtain a dictionary containing:
        * figures (list of (title, Figure) tuples)
        * pandas DataFrames (raw / aggregated / summary)
        * misc dictionaries (trend data, effect histogram, …)
    """

    def __init__(
        self,
        # --------------------- file locations
        basin_shp: str,
        river_shp: str,
        fault_shp: str,
        out_dir: str = "output",
        # --------------------- column names
        distance_col_candidates: Optional[List[str]] = None,
        elevation_col: str = "ELEV_1",
        velocity_col: str = "VEL_1",
        # --------------------- analysis settings
        base_window_km: float = 1.0,
        effect_threshold_mm_yr: float = 0.5,
        min_valid_points: int = 20,
        distance_tolerance_m: float = 500,
        background_resampling_km: Optional[float] = 0.5,
    ) -> None:
        """Store configuration and set up the output directory."""
        # ── Paths ─────────────────────────────────────────────
        self.basin_path = Path(basin_shp)
        self.river_path = Path(river_shp)
        self.fault_path = Path(fault_shp)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # ── Column names ─────────────────────────────────────
        self.distance_candidates = distance_col_candidates or [
            "DIST_UP_KM",
            "DIST_DN_KM",
            "distance",
        ]
        self.elev_col = elevation_col
        self.vel_col = velocity_col

        # ── Analysis parameters ───────────────────────────────
        self.base_window_km = base_window_km
        self.effect_thr = effect_threshold_mm_yr
        self.min_valid_pts = min_valid_points
        self.distance_tolerance_m = distance_tolerance_m
        self.resampling_km = background_resampling_km

        # ── Place‑holders (these are filled later) ───────────────────────
        self.basin: Optional[gpd.GeoDataFrame] = None
        self.river: Optional[gpd.GeoDataFrame] = None
        self.faults: Optional[gpd.GeoDataFrame] = None
        self.distance_col: Optional[str] = None

        self.figures: List[Tuple[str, Any]] = []
        self.stats: Optional[pd.DataFrame] = None
        self.geom_corr: Optional[pd.DataFrame] = None
        self.aggregated: Optional[pd.DataFrame] = None
        self.summary: Optional[Dict[str, Any]] = None

        log.info("RiverFaultAnalyser initialised")

    # -----------------------------------------------------------------
    # 5.1  Data loading & basic preprocessing
    # -----------------------------------------------------------------

    def _load_data(self) -> None:
        """Read the three shapefiles (CRS is forced to EPSG:32646)."""
        log.info("Loading shapefiles…")
        try:
            self.basin = gpd.read_file(self.basin_path).to_crs("EPSG:32646")
            self.river = gpd.read_file(self.river_path).to_crs(self.basin.crs)
            self.faults = gpd.read_file(self.fault_path).to_crs(self.basin.crs)
            self.basin = add_drainage_fields(self.basin)

        except Exception as exc:
            log.exception("Failed to read shapefiles: %s", exc)
            raise

        # Clip faults to the basin polygon
        self.faults = gpd.clip(self.faults, self.basin)
        log.info(
            "Loaded: basin %d, river %d points, faults %d",
            len(self.basin),
            len(self.river),
            len(self.faults),
        )

    def _pick_longest_river(self) -> pd.DataFrame:
        """Select the river with the highest (length * log(Npts)) score."""
        log.info("Choosing the longest, best‑sampled river…")
        if "MAIN_RIV" not in self.river.columns:
            raise KeyError("Column 'MAIN_RIV' missing from river layer.")

        groups = self.river.groupby("MAIN_RIV")
        stats_df = (
        groups["DIST_DN_KM"]
            .agg(["min", "max", "count"])
            .rename(columns={"min": "dist_min", "max": "dist_max", "count": "n_pts"})
        )
        stats_df["length"] = stats_df["dist_max"] - stats_df["dist_min"]
        stats_df["score"] = stats_df["length"] * np.log(stats_df["n_pts"])

        best_river_id = stats_df["score"].idxmax()
        log.info(
            "Selected river %s (length %.2f km, %d points)",
            best_river_id,
            stats_df.loc[best_river_id, "length"],
            stats_df.loc[best_river_id, "n_pts"],
        )
        return self.river[self.river["MAIN_RIV"] == best_river_id].copy()

    def _choose_distance_column(self, df: pd.DataFrame) -> str:
        """
        Pick the distance column with the largest numeric range.
        The chosen column name is stored as `self.distance_col.
        """
        best_range = -1.0
        best_col = None

        for col in self.distance_candidates:
            if col in df.columns:
                r = df[col].max() - df[col].min()
                log.info("Distance column %s: range %.2f km", col, r)
                if r > best_range:
                    best_range = r
                    best_col = col

        if best_col is None or best_range < 0.1:
            raise RuntimeError("No suitable distance column found.")
        self.distance_col = best_col
        return best_col

    def _add_headwater_distance(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create a uniform column `distance_from_headwater_km with
        head‑water (up‑stream) distances increasing downstream.
        """
        if self.distance_col is None:
            raise RuntimeError("Distance column must be chosen first.")
        col = self.distance_col

        if col == "DIST_UP_KM":
            df["distance_from_headwater_km"] = df[col]
        elif col == "DIST_DN_KM":
            max_val = df[col].max()
            df["distance_from_headwater_km"] = max_val - df[col]
        else:
            # Assume the column already contains kilometres, but if >1000 treat as metres.
            if df[col].max() > 1000:
                df["distance_from_headwater_km"] = df[col] / 1000.0
            else:
                df["distance_from_headwater_km"] = df[col]

        # Sort for downstream progression
        return df.sort_values("distance_from_headwater_km").reset_index(drop=True)

    # -----------------------------------------------------------------
    # 5.2  Channel metrics (slope & roughness)
    # -----------------------------------------------------------------

    @staticmethod
    def _smooth_channel(
        distances: np.ndarray, elevations: np.ndarray, window_km: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Thin wrapper around `compute_slope_roughness."""
        return compute_slope_roughness(
            distances=distances, elevations=elevations, window_km=window_km
        )

    def _compute_channel_metrics(
        self,
        df: gpd.GeoDataFrame,
        distance_col: str = "distance_from_headwater_km",
        elevation_col: str = "ELEV_1",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return 1‑D arrays of slope (m/km) and roughness (m).  Existing
        columns are used when present, otherwise they are calculated.
        """
        if {"SLOPE_1", "ROUGH_1"} <= set(df.columns):
            log.info("Using existing slope / roughness columns from river layer")
            slopes = df["SLOPE_1"].values
            rough = df["ROUGH_1"].values
        else:
            log.info("Computing slope / roughness from elevation.")
            distances = df[distance_col].to_numpy()
            elevations = df[elevation_col].to_numpy()
            slopes, rough = self._smooth_channel(
                distances, elevations, window_km=self.base_window_km
            )

        # Fill any NaNs
        slopes = fill_nan(pd.Series(slopes), method="ffill")
        rough = fill_nan(pd.Series(rough), method="ffill")
        return slopes, rough

    # -----------------------------------------------------------------
    # 5.3  Fault‑river intersection
    # -----------------------------------------------------------------

    def _identify_fault_crossings(
        self,
        river_df: gpd.GeoDataFrame,
        fault_gdf: gpd.GeoDataFrame,
    ) -> pd.DataFrame:
        """
        Spatial join – keep only faults that intersect a buffered river centre‑line
        (`distance_tolerance_m) and record the closest river point’s
        distance‑from‑headwater and velocity.
        """
        # Build a centre‑line from the ordered river points
        line = LineString([pt.coords[0] for pt in river_df.geometry])
        buffer = line.buffer(self.distance_tolerance_m)

        # Quick spatial filter
        near_faults = fault_gdf[fault_gdf.geometry.intersects(buffer)].copy()
        log.info("Found %d faults within %.0f m of the river", len(near_faults), self.distance_tolerance_m)

        records = []
        for idx, fault in near_faults.iterrows():
            # Distance to every river point (this process is quite fast thanks to vectorisation)
            dists = river_df.geometry.distance(fault.geometry)
            closest_idx = dists.idxmin()
            if dists.iloc[closest_idx] > self.distance_tolerance_m:
                continue  # safety – this should not happen due to the buffer test!

            row = river_df.loc[closest_idx]
            records.append({
                "fault_idx": idx,
                "fault_type": fault["Fea_En_Clean"],
                "distance_km": row["distance_from_headwater_km"],
                "velocity_mm_yr": row[self.vel_col],
                "landcover_class": row["LandCover_"],
                "lc_meta": recode_landcover(row["LandCover_"]),  
            })


        df = pd.DataFrame(records)
        log.info("Identified %d fault–river crossings.", len(df))
        return df


    # -----------------------------------------------------------------
    # 5.4b  Permutation test – global null for “significant” counts
    # -----------------------------------------------------------------
    def _permutation_test_overall(
        self,
        distances: np.ndarray,
        residuals: np.ndarray,
        n_crossings: int,
        effect_thr: float,
        window_km: float,
        n_perm: int = 200,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Empirical null: randomly place n_crossings 'fake' centers along the river,
        compute mean-based effect & Welch t-test with the SAME window size,
        count how many pass (|Δ| > effect_thr AND p_raw < 0.05).
        Returns the null distribution and summary stats.
        """
        rng = np.random.default_rng(random_state)
        L = distances
        counts = np.empty(n_perm, dtype=int)

        for i in range(n_perm):
            fake_centers = rng.choice(L, size=min(n_crossings, len(L)), replace=False)
            cnt = 0
            for f in fake_centers:
                up = (L >= f - window_km) & (L < f)
                dn = (L > f) & (L <= f + window_km)
                if up.sum() < 3 or dn.sum() < 3:
                    continue
                eff = float(np.mean(residuals[dn]) - np.mean(residuals[up]))
                _, p = stats.ttest_ind(residuals[up], residuals[dn], equal_var=False)
                if (abs(eff) > effect_thr) and (p < 0.05):
                    cnt += 1
            counts[i] = cnt

        return {
            "null_counts": counts,
            "null_mean": float(np.mean(counts)),
            "null_std": float(np.std(counts, ddof=1) if len(counts) > 1 else 0.0),
            "null_q95": float(np.quantile(counts, 0.95)),
        }
    
    def _write_permutation_summary(self, out_prefix: str, obs_count: int, perm: Dict[str, Any], n_perm: int) -> None:
        df = pd.DataFrame([{
            "obs_truly_like_count": obs_count,
            "null_mean": perm["null_mean"],
            "null_std": perm["null_std"],
            "null_q95": perm["null_q95"],
            "n_perm": n_perm,
        }])
        df.to_csv(self.out_dir / f"{out_prefix}_permutation_summary.csv", index=False)
        log.info("Saved permutation summary CSV: %s", f"{out_prefix}_permutation_summary.csv")

    # -----------------------------------------------------------------
    # 5.4  High‑resolution fault‑by‑fault analysis
    # -----------------------------------------------------------------

    def _highres_analysis(
        self,
        river_df: gpd.GeoDataFrame,
        crossings: pd.DataFrame,
        slopes: np.ndarray,
    ) -> Tuple[pd.DataFrame, Dict, Dict, np.ndarray, np.ndarray]:
        """
        Core analysis – for each (consolidated) fault crossing:

        * build a background linear trend (velocity vs distance)
        * calculate upstream / downstream windows (adaptive size)
        * compute raw and residual effect sizes
        * perform a two‑sample t‑test on the residuals
        * collect p‑values for later multiple‑testing correction
        """
        # -----------------------------------------------------------------
        # 1️)  Background trend
        # -----------------------------------------------------------------
        distances = river_df["distance_from_headwater_km"].to_numpy()
        velocities = river_df[self.vel_col].to_numpy()
        mask = ~np.isnan(distances) & ~np.isnan(velocities)
        distances = distances[mask]
        velocities = velocities[mask]
        slopes = slopes[mask]


        if len(distances) < self.min_valid_pts:
            raise RuntimeError(
                f"Not enough valid river points (need ≥{self.min_valid_pts})"
            )

        trend_info = fit_background_trend(distances, velocities)
        vel_residuals = trend_info["residuals"]
        residual_std = trend_info["residual_std"]

        # --- Alternate detrending (LOWESS) for robustness ---
        lowess_info = fit_background_trend_lowess(distances, velocities, frac=0.2)
        vel_residuals_lowess = lowess_info["residuals"]


        # -----------------------------------------------------------------
        # 2️)  Consolidate duplicate faults (the purpose of this is to help with very dense crossing lists)
        # -----------------------------------------------------------------
        consolidated = consolidate_duplicate_faults(crossings, distance_tolerance_km=0.5)

        # -----------------------------------------------------------------
        # 3️)  Loop over faults, compute effect sizes, collect p‑values
        # -----------------------------------------------------------------
        results: List[Dict[str, Any]] = []
        raw_p_vals: List[float] = []

        for _, fault in consolidated.iterrows():
            f_dist = fault["distance_km"]
            f_type = fault["fault_type"]

            # Adaptive window based on local slope variability
            local_mask = np.abs(distances - f_dist) <= 2 * self.base_window_km
            window_km = (
                adaptive_window_size(slopes[local_mask], base_window_km=self.base_window_km)
                if np.any(local_mask)
                else self.base_window_km
            )

            # Up‑stream & down‑stream masks
            upstream = (distances >= f_dist - window_km) & (distances < f_dist)
            downstream = (distances > f_dist) & (distances <= f_dist + window_km)

            if upstream.sum() < 3 or downstream.sum() < 3:
                # Not enough points for a reliable test → skip
                continue

            # Raw effect (difference of means) and residual effect
            raw_eff = np.mean(velocities[downstream]) - np.mean(velocities[upstream])
            resid_eff = np.mean(vel_residuals[downstream]) - np.mean(
                vel_residuals[upstream]
            )

            # Two‑sample t‑test on the *residuals*
            t_stat, p_raw = stats.ttest_ind(
                vel_residuals[upstream], vel_residuals[downstream], equal_var=False
            )

            # Robust (median) effect & non-parametric test on linear residuals
            median_eff = float(np.median(vel_residuals[downstream]) - np.median(vel_residuals[upstream]))
            _, p_mw = stats.mannwhitneyu(vel_residuals[upstream], vel_residuals[downstream], alternative="two-sided")

            # Do the same under LOWESS residuals (alt detrending)
            median_eff_lowess = float(np.median(vel_residuals_lowess[downstream]) - np.median(vel_residuals_lowess[upstream]))
            _, p_mw_lowess = stats.mannwhitneyu(vel_residuals_lowess[upstream], vel_residuals_lowess[downstream], alternative="two-sided")


            raw_p_vals.append(p_raw)

            # Land-cover proportions inside the same test window we use for Δ
            lc_props = lc_window_proportions(
                river_df=river_df,
                center_km=f_dist,
                window_km=window_km,
                dist_col="distance_from_headwater_km",
                lc_col="LandCover_",
            )

            basin_drain = str(self.basin.iloc[0].get("drainage_class", "unknown"))
            basin_is_internal = bool(self.basin.iloc[0].get("is_internal", False))
            lc_point_meta = recode_landcover(fault.get("landcover_class", None))
            lc_mode = lc_props.get("lc_mode", "Unknown")
            lc_entropy = lc_props.get("lc_entropy", np.nan)



            results.append(
                {
                    "fault_distance_km": f_dist,
                    "fault_type": f_type,
                    "landcover_class": fault["landcover_class"],
                    "upstream_n": int(upstream.sum()),
                    "downstream_n": int(downstream.sum()),
                    "raw_effect_mm_yr": raw_eff,
                    "residual_effect_mm_yr": resid_eff,
                    "t_statistic": t_stat,
                    "p_value_raw": p_raw,
                    "window_km_used": window_km,
                    "lc_mode": lc_props.get("lc_mode", "Unknown"),
                    "lc_entropy": lc_props.get("lc_entropy", np.nan),
                    "lc_point_meta": lc_point_meta,
                    **{k: v for k, v in lc_props.items() if k.startswith("p_lc_")},
                    "drainage_class": basin_drain,
                    "is_internal": basin_is_internal,
                    "n_faults_consolidated": fault["n_faults_consolidated"],
                    "residual_effect_median_mm_yr": median_eff,
                    "p_value_mw": p_mw,
                    "robust_significant": (abs(median_eff) > self.effect_thr) and (p_mw < 0.05),
                    "residual_effect_median_lowess_mm_yr": median_eff_lowess,
                    "p_value_mw_lowess": p_mw_lowess,
                    "robust_significant_lowess": (abs(median_eff_lowess) > self.effect_thr) and (p_mw_lowess < 0.05),
                    "effect_significant": abs(raw_eff) > self.effect_thr,
                }
            )

        # -----------------------------------------------------------------
        # 4️)  Multiple‑testing correction
        # -----------------------------------------------------------------
        if raw_p_vals:
            corrections = multiple_testing_correction(np.array(raw_p_vals), alpha=0.05)
            for i, row in enumerate(results):
                row["p_value_fdr"] = corrections["p_fdr"][i]
                row["p_value_bonf"] = corrections["p_bonf"][i]
                row["significant_fdr"] = corrections["fdr_rej"][i]
                row["significant_bonf"] = corrections["bonf_rej"][i]
                row["statistically_significant"] = row["p_value_raw"] < 0.05
                row["truly_significant"] = row["effect_significant"] and row[
                    "significant_fdr"
                ]
        else:
            log.warning("No faults passed the data‑availability filter – no statistics computed.")

        results_df = pd.DataFrame(results)

        # -----------------------------------------------------------------
        # 5️)  Assemble auxiliary objects needed for plotting
        # -----------------------------------------------------------------
        effect_hist = {
            "all_effects": np.abs(results_df["raw_effect_mm_yr"].values)
            if not results_df.empty
            else np.array([]),
            "threshold": self.effect_thr,
            "n_above_threshold": int(
                (np.abs(results_df["raw_effect_mm_yr"].values) > self.effect_thr).sum()
            ),
        }

        trend_data = {
            "distances": distances,
            "velocities": velocities,
            "background_trend": trend_info["trend"],
            "velocity_residuals": vel_residuals,
            "slope": trend_info["slope"],
            "intercept": trend_info["intercept"],
            "r_value": trend_info["r"],
            "p_value": trend_info["p"],
            "std_err": trend_info["std_err"],
            "residual_std": residual_std,
        }

        return results_df, effect_hist, trend_data, distances, velocities

    # -----------------------------------------------------------------
    # 5.5  Geomorphic correlation checks (slope & roughness)
    # -----------------------------------------------------------------

    def _geomorphic_correlation_check(
        self,
        river_df: gpd.GeoDataFrame,
        statistical_results: pd.DataFrame,
        slopes: np.ndarray,
        roughness: np.ndarray,
        distance_col: str = "distance_from_headwater_km",
        show_profiles: bool = True,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        For each *truly* significant fault, test whether there is also a
        statistically significant change in (a) channel slope or (b) roughness.
        Returns a summary DataFrame and a list of raw profile windows
        (useful for custom plots).
        """
        distances = river_df[distance_col].values
        truly_sig = statistical_results[
            statistical_results["truly_significant"]
        ].copy()

        corr_results: List[Dict[str, Any]] = []
        profiles: List[Dict[str, Any]] = []

        for _, fault in truly_sig.iterrows():
            f_dist = fault["fault_distance_km"]
            f_type = fault["fault_type"]
            win = fault["window_km_used"]

            # Up‑ and downstream masks (the same windows used for the velocity test)
            up_mask = (distances >= f_dist - win) & (distances < f_dist)
            down_mask = (distances > f_dist) & (distances <= f_dist + win)

            # --- Slope test -------------------------------------------------
            if up_mask.sum() >= 3 and down_mask.sum() >= 3:
                s_up = slopes[up_mask]
                s_down = slopes[down_mask]
                slope_stat, slope_p = stats.mannwhitneyu(
                    s_up, s_down, alternative="two-sided"
                )
                slope_effect = np.median(s_down) - np.median(s_up)
            else:
                slope_p, slope_effect = 1.0, 0.0

            # --- Roughness test --------------------------------------------
            if up_mask.sum() >= 3 and down_mask.sum() >= 3:
                r_up = roughness[up_mask]
                r_down = roughness[down_mask]
                rough_stat, rough_p = stats.mannwhitneyu(
                    r_up, r_down, alternative="two-sided"
                )
                rough_effect = np.median(r_down) - np.median(r_up)
            else:
                rough_p, rough_effect = 1.0, 0.0

            corr_results.append(
                {
                    "fault_distance_km": f_dist,
                    "fault_type": f_type,
                    "velocity_effect_mm_yr": fault["raw_effect_mm_yr"],
                    "slope_effect_m_km": slope_effect,
                    "slope_p_value": slope_p,
                    "roughness_effect_m": rough_effect,
                    "roughness_p_value": rough_p,
                    "slope_significant": slope_p < 0.05,
                    "roughness_significant": rough_p < 0.05,
                    "any_geomorphic_change": (slope_p < 0.05) or (rough_p < 0.05),
                }
            )

            if show_profiles:
                profiles.append(
                    {
                        "fault_distance_km": f_dist,
                        "fault_type": f_type,
                        "velocity_effect": fault["raw_effect_mm_yr"],
                        "distances": distances[(distances >= f_dist - 2 * win) & (distances <= f_dist + 2 * win)]
                        - f_dist,
                        "slopes": slopes[(distances >= f_dist - 2 * win) & (distances <= f_dist + 2 * win)],
                        "roughness": roughness[(distances >= f_dist - 2 * win) & (distances <= f_dist + 2 * win)],
                    }
                )

        corr_df = pd.DataFrame(corr_results)
        return corr_df, profiles



    # -----------------------------------------------------------------
    # 5.6  Plotting helpers (each returns a Figure)
    # -----------------------------------------------------------------

    @staticmethod
    def _add_figure(fig: plt.Figure, title: str, fig_list: List[Tuple[str, Any]]) -> None:
        """Utility to keep the figures list synchronised."""
        fig_list.append((title, fig))

    # ---- Figure 1 – background trend -------------------------------------------------
    @staticmethod
    def _create_figure_1(
        trend_data: Dict[str, Any],
        river_id: Any,
        aggregated: Optional[pd.DataFrame] = None,
    ) -> plt.Figure:
        """Trend line + optional binned aggregation."""
        fig, ax = plt.subplots(figsize=(14, 8))

        if aggregated is not None:
            ax.errorbar(
                aggregated["distance_km"],
                aggregated["velocity_mean"],
                yerr=aggregated["velocity_std"],
                fmt="o",
                markersize=4,
                alpha=0.7,
                capsize=3,
                label="Aggregated InSAR velocities",
            )
            xs = aggregated["distance_km"].values
        else:
            # Sub‑sample for visual cleanliness when there are many points
            if len(trend_data["distances"]) > 2000:
                ids = np.random.choice(
                    len(trend_data["distances"]), 2000, replace=False
                )
                ids = np.sort(ids)
                xs = trend_data["distances"][ids]
                ax.scatter(
                    xs,
                    trend_data["velocities"][ids],
                    c="lightblue",
                    s=8,
                    alpha=0.6,
                    label="Raw InSAR velocities",
                    edgecolors="none",
                )
            else:
                xs = trend_data["distances"]
                ax.scatter(
                    xs,
                    trend_data["velocities"],
                    c="lightblue",
                    s=8,
                    alpha=0.6,
                    label="Raw InSAR velocities",
                    edgecolors="none",
                )

        # Trend line & 95 % confidence band
        trend_line = trend_data["slope"] * xs + trend_data["intercept"]
        ax.plot(xs, trend_line, "r-", linewidth=3, label="Background trend")

        ci = 1.96 * trend_data["residual_std"]
        ax.fill_between(
            xs,
            trend_line - ci,
            trend_line + ci,
            color="red",
            alpha=0.2,
            label="95 % confidence interval",
        )

        ax.set_xlabel("Distance from headwater (km)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Vertical velocity (mm yr⁻¹)", fontsize=14, fontweight="bold")
        ax.set_title(
            f"Basin 1 – Regional velocity trend (River {river_id})",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc="lower right")
        plt.tight_layout()
        return fig

    # ---- Figure 2 – multi‑panel longitudinal profiles ---------------------------------
    @staticmethod
    def _create_figure_2(
        trend_data: Dict[str, Any],
        slopes: np.ndarray,
        roughness: np.ndarray,
        statistical_results: pd.DataFrame,
        river_id: Any,
        aggregated: Optional[pd.DataFrame] = None,
    ) -> plt.Figure:
        """Three‑panel plot (detrended velocity, slope, roughness) with fault markers."""
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(16, 12), sharex=True
        )

        # -------------------------------------------------
        # Panel 1 – detrended velocity
        # -------------------------------------------------
        if aggregated is not None:
            ax1.errorbar(
                aggregated["distance_km"],
                aggregated["velocity_mean"],
                yerr=aggregated["velocity_std"],
                fmt="o",
                markersize=4,
                alpha=0.7,
                capsize=3,
                label="Aggregated velocities",
            )
            xs = aggregated["distance_km"].values
            vel_plot = aggregated["velocity_mean"].values
        else:
            xs = trend_data["distances"]
            resid = trend_data["velocity_residuals"]
            # --- Fix for x/y mismatch ---
            if len(xs) != len(resid):
                minlen = min(len(xs), len(resid))
                print(f"[WARN] Length mismatch in plotting: xs={len(xs)}, resid={len(resid)}. Truncating to {minlen}")
                xs = xs[:minlen]
                resid = resid[:minlen]
            ax1.scatter(xs, resid, c="steelblue", s=10, alpha=0.7, label="Detrended velocity")

        ax1.axhline(0, color="black", linewidth=1, alpha=0.5)
        ax1.set_ylabel("Detrended velocity (mm yr⁻¹)", fontsize=12, fontweight="bold")
        ax1.set_title(
            f"Basin 1 – River {river_id} longitudinal profiles", fontsize=14
        )

        # ----- Panel 2 – Slope -----
        if aggregated is not None:
            x_slopes = aggregated["distance_km"].values
            y_slopes = aggregated["slope_mean"].values
        else:
            x_slopes = xs
            y_slopes = slopes
        if len(x_slopes) != len(y_slopes):
            minlen = min(len(x_slopes), len(y_slopes))
            print(f"[WARN] Slope: Length mismatch: x={len(x_slopes)}, y={len(y_slopes)}. Truncating to {minlen}")
            x_slopes = x_slopes[:minlen]
            y_slopes = y_slopes[:minlen]
        ax2.plot(x_slopes, y_slopes, "g-", linewidth=2, label="Channel slope")
        ax2.fill_between(x_slopes, y_slopes, alpha=0.3, color="g")

        # ----- Panel 3 – Roughness -----
        if aggregated is not None:
            x_rough = aggregated["distance_km"].values
            y_rough = aggregated["roughness_mean"].values
        else:
            x_rough = xs
            y_rough = roughness
        if len(x_rough) != len(y_rough):
            minlen = min(len(x_rough), len(y_rough))
            print(f"[WARN] Roughness: Length mismatch: x={len(x_rough)}, y={len(y_rough)}. Truncating to {minlen}")
            x_rough = x_rough[:minlen]
            y_rough = y_rough[:minlen]
        ax3.plot(x_rough, y_rough, "brown", linewidth=2, label="Channel roughness")
        ax3.fill_between(x_rough, y_rough, alpha=0.3, color="brown")


        # -------------------------------------------------
        # Fault markers (only those that are *truly* significant faults)
        # -------------------------------------------------
        if not statistical_results.empty:
            sig = statistical_results[statistical_results["truly_significant"]]
            # Use a single colour per fault type
            fault_cols = {
                "Reverse": "crimson",
                "Normal": "blue",
                "Left Lateral": "green",
                "Right Lateral": "purple",
                "Strike‑Slip": "orange",
                "Buried": "gray",
                "Unknown": "black",
            }
            plotted = set()
            for _, row in sig.iterrows():
                col = fault_cols.get(row["fault_type"], "black")
                label = row["fault_type"] if row["fault_type"] not in plotted else ""
                for ax in (ax1, ax2, ax3):
                    ax.axvline(
                        row["fault_distance_km"],
                        color=col,
                        linewidth=2,
                        alpha=0.9,
                        label=label,
                    )
                plotted.add(row["fault_type"])

        for ax in (ax1, ax2, ax3):
            ax.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc="upper right")
        ax2.legend(fontsize=10, loc="upper right")
        ax3.legend(fontsize=10, loc="upper right")
        plt.tight_layout()
        return fig

    # ---- Figure 3 – effect‑size histogram --------------------------------------------
    @staticmethod
    def _create_figure_3(
        effect_hist: Dict[str, Any], statistical_results: pd.DataFrame
    ) -> plt.Figure:
        """Histogram of absolute effect sizes with the significance threshold."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        effects = effect_hist["all_effects"]
        thr = effect_hist["threshold"]

        # ---- Left panel – histogram -------------------------------------------------
        n_bins = 25
        counts, bins, patches = ax1.hist(
            effects, bins=n_bins, edgecolor="black", alpha=0.8
        )
        bin_centers = (bins[:-1] + bins[1:]) / 2
        for patch, centre in zip(patches, bin_centers):
            if centre >= thr:
                patch.set_facecolor("crimson")
                patch.set_alpha(0.7)
            else:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)

        ax1.axvline(thr, color="darkred", linestyle="--", linewidth=3, label="Threshold")
        ax1.axvline(np.median(effects), color="darkgreen", linestyle="-", linewidth=3, label="Median")
        ax1.set_xlabel("|Δ velocity| (mm yr⁻¹)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Number of faults", fontsize=12, fontweight="bold")
        ax1.set_title("Effect‑size distribution", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ---- Right panel – correction summary ---------------------------------------
        if not statistical_results.empty:
            corr = {
                "Raw p<0.05": (statistical_results["p_value_raw"] < 0.05).sum(),
                "FDR‑corrected": statistical_results["significant_fdr"].sum(),
                "Bonferroni": statistical_results["significant_bonf"].sum(),
                "Truly significant (effect + FDR)": statistical_results["truly_significant"].sum(),
            }

            cats = list(corr.keys())
            vals = list(corr.values())
            colors = ["lightcoral", "orange", "lightblue", "darkgreen"]
            bars = ax2.bar(cats, vals, color=colors, edgecolor="black", alpha=0.8)
            for bar, val in zip(bars, vals):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
            ax2.set_ylabel("Number of faults", fontsize=12, fontweight="bold")
            ax2.set_title("Impact of multiple‑testing corrections", fontsize=14, fontweight="bold")
            ax2.grid(True, axis="y", alpha=0.3)
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        else:
            ax2.text(0.5, 0.5, "No statistical results", ha="center", va="center")

        plt.suptitle("Basin 1 – Effect‑size analysis & statistical corrections", fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    # ---- Figure 4 – volcano plot ----------------------------------------------------
    @staticmethod
    def _create_figure_4(statistical_results: pd.DataFrame, effect_thr: float) -> plt.Figure:

        """Two‑panel volcano plot (raw p‑values on the left, FDR‑corrected on the right)."""
        if statistical_results.empty:
            raise ValueError("No statistical results to plot.")

        fig, (ax_raw, ax_fdr) = plt.subplots(1, 2, figsize=(18, 8))

        # Symbol / colour map per fault type
        symb_col = {
            "Reverse": ("^", "crimson"),
            "Normal": ("v", "blue"),
            "Left Lateral": ("<", "green"),
            "Right Lateral": (">", "purple"),
            "Strike‑Slip": ("s", "orange"),
            "Buried": ("o", "gray"),
            "Unknown": ("X", "black"),
        }

        # ---- Left panel – raw p‑values -------------------------------------------
        plotted = set()
        for _, row in statistical_results.iterrows():
            sym, col = symb_col.get(row["fault_type"], ("o", "black"))
            size = 100 if row["truly_significant"] else (70 if row["effect_significant"] else 40)
            alpha = 1.0 if row["truly_significant"] else (0.8 if row["effect_significant"] else 0.5)
            label = row["fault_type"] if row["fault_type"] not in plotted else ""
            ax_raw.scatter(
                row["raw_effect_mm_yr"],
                -np.log10(row["p_value_raw"]),
                s=size,
                c=col,
                marker=sym,
                alpha=alpha,
                edgecolors="black",
                linewidths=1,
                label=label,
            )
            plotted.add(row["fault_type"])

        ax_raw.axhline(-np.log10(0.05), color="blue", linestyle="--", linewidth=2, alpha=0.7)
        ax_raw.axhline(-np.log10(0.01), color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax_raw.axvline(effect_thr, color="darkred", linestyle=":", linewidth=2, alpha=0.7)
        ax_raw.axvline(-effect_thr, color="darkred", linestyle=":", linewidth=2, alpha=0.7)
        ax_raw.set_xlabel("Effect size (mm yr⁻¹)\n← Subsidence | Uplift →", fontsize=12, fontweight="bold")
        ax_raw.set_ylabel("-log₁₀(p‑value)", fontsize=12, fontweight="bold")
        ax_raw.set_title("Raw p‑values", fontsize=14, fontweight="bold")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.legend(fontsize=10, loc="upper left", framealpha=0.9)

        # ---- Right panel – FDR‑corrected p‑values ---------------------------------
        plotted = set()
        for _, row in statistical_results.iterrows():
            sym, col = symb_col.get(row["fault_type"], ("o", "black"))
            size = 100 if row["truly_significant"] else (70 if row["significant_fdr"] else 40)
            alpha = 1.0 if row["truly_significant"] else (0.8 if row["significant_fdr"] else 0.5)
            label = row["fault_type"] if row["fault_type"] not in plotted else ""
            ax_fdr.scatter(
                row["raw_effect_mm_yr"],
                -np.log10(row["p_value_fdr"]),
                s=size,
                c=col,
                marker=sym,
                alpha=alpha,
                edgecolors="black",
                linewidths=1,
                label=label,
            )
            plotted.add(row["fault_type"])

        ax_fdr.axhline(-np.log10(0.05), color="blue", linestyle="--", linewidth=2, alpha=0.7)
        ax_fdr.axhline(-np.log10(0.01), color="red", linestyle="--", linewidth=2, alpha=0.7)
        ax_fdr.axvline(effect_thr,  color="darkred", linestyle=":", linewidth=2, alpha=0.7)
        ax_fdr.axvline(-effect_thr, color="darkred", linestyle=":", linewidth=2, alpha=0.7)
        ax_fdr.set_xlabel("Effect size (mm yr⁻¹)\n← Subsidence | Uplift →", fontsize=12, fontweight="bold")
        ax_fdr.set_ylabel("-log₁₀(p‑value)", fontsize=12, fontweight="bold")
        ax_fdr.set_title("FDR‑corrected p‑values", fontsize=14, fontweight="bold")
        ax_fdr.grid(True, alpha=0.3)
        ax_fdr.legend(fontsize=10, loc="upper left", framealpha=0.9)

        plt.suptitle("Basin 1 – Volcano plots (raw vs. FDR‑corrected)", fontsize=16, fontweight="bold")
        plt.tight_layout()
        return fig

    # ---- Figure 5 – spatial context ------------------------------------------------
    @staticmethod
    def _create_figure_5(
        river_df: gpd.GeoDataFrame,
        distance_col: str,
        statistical_results: pd.DataFrame,
        basin_gdf: gpd.GeoDataFrame,
        faults_gdf: gpd.GeoDataFrame,         
        river_id: Any,
    ) -> Optional[plt.Figure]:
        """Map of the basin, river, and only the *truly* significant faults."""
        sig = statistical_results[statistical_results["truly_significant"]]
        if "truly_significant" in statistical_results.columns:
            sig = statistical_results.loc[statistical_results["truly_significant"]].copy()
        else:
            sig = pd.DataFrame()
        if sig.empty:
            log.info("No truly significant faults – skipping Figure 5.")
            return None

        fig, ax = plt.subplots(figsize=(14, 10))

        # Basin boundary
        basin_gdf.plot(ax=ax, facecolor="lightblue", edgecolor="navy", linewidth=2, alpha=0.4, label="Basin")

        # River (subsampled for visual clarity)
        river_df.iloc[::10].plot(ax=ax, color="blue", linewidth=0.8, label="River")

        # ---- Fault traces (clipped to basin, coloured by type) -----------------
        fault_cols = {
            "Reverse": "crimson",
            "Normal": "blue",
            "Left Lateral": "green",
            "Right Lateral": "purple",
            "Strike-Slip": "orange",
            "Buried": "gray",
            "Unknown": "black",
        }

        # This makes sure the cleaned fault-type column exists 
        if "Fea_En_Clean" not in faults_gdf.columns:
            faults_plot = faults_gdf.copy()
            colname = "Fea_En" if "Fea_En" in faults_plot.columns else None
            faults_plot["Fea_En_Clean"] = (
                faults_plot[colname].astype(str) if colname else "Unknown"
            )
        else:
            faults_plot = faults_gdf

        # Draw each fault-type in its own colour
        for ftype, col in fault_cols.items():
            sub = faults_plot[faults_plot["Fea_En_Clean"] == ftype]
            if len(sub):
                # thin line behind markers, above basin fill
                sub.plot(ax=ax, color=col, linewidth=1.4, alpha=0.9, zorder=6, label=ftype)


        plotted = set()
        for _, row in sig.iterrows():
            # Find the nearest river point (this gets the xy coordinates)
            dists = river_df[distance_col].values - row["fault_distance_km"]
            nearest_idx = np.argmin(np.abs(dists))
            pt = river_df.iloc[nearest_idx].geometry
            col = fault_cols.get(row["fault_type"], "black")
            size = 100 + np.abs(row["raw_effect_mm_yr"]) * 200

            label = row["fault_type"] if row["fault_type"] not in plotted else ""
            ax.scatter(
                pt.x,
                pt.y,
                s=size,
                c=col,
                marker="*",
                edgecolors="white",
                linewidths=2,
                alpha=0.9,
                label=label,
                zorder=10,
            )
            plotted.add(row["fault_type"])

            # Annotate the strongest few (|Δ| > 0.8 mm/yr)
            if np.abs(row["raw_effect_mm_yr"]) > 0.8:
                ax.annotate(
                    f"{row['fault_distance_km']:.1f} km\nΔ={row['raw_effect_mm_yr']:.2f}",
                    xy=(pt.x, pt.y),
                    xytext=(20, 20),
                    textcoords="offset points",
                    fontsize=10,
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        edgecolor=col,
                        alpha=0.9,
                    ),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
                )

        ax.set_xlabel("Easting (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Northing (m)", fontsize=12, fontweight="bold")
        ax.set_title(
            f"Basin 1 – Spatial distribution of significant faults (River {river_id})",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        plt.tight_layout()
        return fig

    # --- ADD: Figure 6 – summaries by drainage & dominant LC at crossings ---
    def _create_figure_6(self, statistical_results: pd.DataFrame) -> Optional[plt.Figure]:
        """Bar charts: drainage class and dominant land-cover for truly significant jumps."""
        sig = statistical_results[statistical_results.get("truly_significant", False)].copy()
        if "truly_significant" in statistical_results.columns:
            sig = statistical_results.loc[statistical_results["truly_significant"]].copy()
        else:
            sig = pd.DataFrame()

        if sig.empty:
            log.info("No truly significant crossings – skipping Figure 6.")
            return None

        import numpy as np
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=False)
        fig.suptitle("Basin 1 – Drainage & land-cover context of significant jumps", fontsize=16, fontweight="bold")

        # --- Panel A: drainage class counts ---
        a = (sig["drainage_class"]
            .fillna("unknown")
            .value_counts(dropna=False)
            .rename_axis("class")
            .reset_index(name="n"))
        if "class" not in a.columns:  # safety for older pandas patterns
            a = a.rename(columns={a.columns[0]: "class"})
        ax1.bar(a["class"], a["n"], edgecolor="black", alpha=0.85)
        ax1.set_title("Significant crossings by drainage class", fontsize=14, fontweight="bold")
        ax1.set_xlabel("")
        ax1.set_ylabel("Count")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=25, ha="right")

        # --- Panel B: dominant land cover within ±window (fallback to point LC if needed) ---
        # ensure recoding is applied to both window mode and fallback
        if "lc_mode" in sig.columns:
            sig["lc_mode"] = sig["lc_mode"].apply(recode_landcover)
        if "lc_point_meta" in sig.columns:
            sig["lc_point_meta"] = sig["lc_point_meta"].apply(recode_landcover)

        sig["lc_mode_fallback"] = np.where(
            sig.get("lc_mode").isin(["Other", "Unknown"]) if "lc_mode" in sig.columns else True,
            sig.get("lc_point_meta", sig.get("landcover_class", "Other")).apply(recode_landcover)
                if "lc_point_meta" in sig.columns or "landcover_class" in sig.columns else "Other",
            sig.get("lc_mode", "Other")
        )

        b = (sig["lc_mode_fallback"]
            .fillna("Other")
            .value_counts(dropna=False)
            .rename_axis("lc")
            .reset_index(name="n"))
        if "lc" not in b.columns:
            b = b.rename(columns={b.columns[0]: "lc"})
        ax2.bar(b["lc"], b["n"], edgecolor="black", alpha=0.85)
        ax2.set_title("Significant crossings by dominant LC (±window)", fontsize=14, fontweight="bold")
        ax2.set_xlabel("")
        ax2.set_ylabel("Count")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=25, ha="right")

        plt.tight_layout()
        return fig
    
    # -----------------------------------------------------------------
# 5.1  CSV exports for drainage & land-cover context
# -----------------------------------------------------------------
    def _export_context_csvs(
        self,
        statistical_results: pd.DataFrame,
        out_prefix: str,
        sig_mask: str = "truly",
    ):
        """
        Write per-crossing context + simple counts even when there are zero rows.
        sig_mask: 'truly' | 'fdr' | 'raw' | 'all'
        """
        # ---- choose the subset ---------------------------------------------
        if sig_mask == "truly":
            sig = statistical_results[statistical_results.get("truly_significant", False)].copy()
            if "truly_significant" in statistical_results.columns:
                sig = statistical_results.loc[statistical_results["truly_significant"]].copy()
            else:
                sig = pd.DataFrame()
        elif sig_mask == "fdr":
            sig = statistical_results[statistical_results.get("significant_fdr", False)].copy()
        elif sig_mask == "raw":
            sig = statistical_results[statistical_results.get("p_value_raw", 1.0) < 0.05].copy()
        elif sig_mask == "all":
            sig = statistical_results.copy()
        else:
            raise ValueError(f"Unknown sig_mask: {sig_mask}")

        # ---- paths ----------------------------------------------------------
        p1 = self.out_dir / f"{out_prefix}_sig_{sig_mask}_context_per_crossing.csv"
        p2 = self.out_dir / f"{out_prefix}_sig_{sig_mask}_counts_by_drainage.csv"
        p3 = self.out_dir / f"{out_prefix}_sig_{sig_mask}_counts_by_landcover.csv"

        # ---- per-crossing table (select only columns that exist) -----------
        wanted = [
            "fault_distance_km","fault_type","raw_effect_mm_yr","p_value_raw","p_value_fdr",
            "truly_significant","significant_fdr","window_km_used","n_faults_consolidated",
            "drainage_class","lc_mode","lc_mode_share","lc_point_meta","LandCover_"
        ]
        cols = [c for c in wanted if c in sig.columns]
        pd.DataFrame(columns=wanted).to_csv(p1, index=False) if sig.empty else sig[cols].to_csv(p1, index=False)

        # ---- counts-----------------------------------------------------------
        if sig.empty:
            log.info("No crossings for mask '%s' – writing empty context CSVs.", sig_mask)
            pd.DataFrame(columns=["drainage_class","n","percent"]).to_csv(p2, index=False)
            pd.DataFrame(columns=["landcover","n","percent"]).to_csv(p3, index=False)
            return {"per_crossing": p1, "counts_by_drainage": p2, "counts_by_landcover": p3}

        # drainage counts
        a = (
            sig.get("drainage_class", pd.Series([], dtype="object"))
            .fillna("Unknown")
            .value_counts(dropna=False)
            .rename_axis("drainage_class")
            .reset_index(name="n")
        )
        a["percent"] = (a["n"] / len(sig) * 100).round(1)
        a.to_csv(p2, index=False)

        # land-cover counts
        b = (
            sig.get("lc_mode", pd.Series([], dtype="object"))
            .fillna("Unknown")
            .value_counts(dropna=False)
            .rename_axis("landcover")
            .reset_index(name="n")
        )
        b["percent"] = (b["n"] / len(sig) * 100).round(1)
        b.to_csv(p3, index=False)

        return {"per_crossing": p1, "counts_by_drainage": p2, "counts_by_landcover": p3}

# -----------------------------------------------------------------
# 5.2  Sensitivity / robustness grid
# -----------------------------------------------------------------
    def _run_sensitivity_grid(
        self,
        river_df: gpd.GeoDataFrame,
        slopes: np.ndarray,
        roughness: np.ndarray,
        out_prefix: str,
        window_km_list: Tuple[float, ...] = (0.5, 1.0, 2.0),
        effect_thresholds: Tuple[float, ...] = (0.3, 0.5, 0.7),
        distance_tolerances_m: Tuple[int, ...] = (250, 500, 1000),
    ) -> pd.DataFrame:
        """
        Re-runs the fault-crossing stats across a small parameter grid and writes a CSV.
        Comparison against window size, effect threshold, and river–fault distance tolerance.
        Returns the DataFrame that is written to disk.
        """
        _orig = dict(
            base_window_km=self.base_window_km,
            effect_thr=self.effect_thr,
            distance_tolerance_m=self.distance_tolerance_m,
        )

        rows: List[Dict[str, Any]] = []
        combos = list(product(window_km_list, effect_thresholds, distance_tolerances_m))
        log.info("Sensitivity grid: %d combinations.", len(combos))

        for win_km, thr, tol_m in combos:
            try:
                # tweak parameters
                self.base_window_km = float(win_km)
                self.effect_thr = float(thr)
                self.distance_tolerance_m = int(tol_m)

                # recompute crossings with new tolerance
                crossings = self._identify_fault_crossings(river_df, self.faults)

                # run high-res analysis
                stat_df, effect_hist, trend_data, dists, vels = self._highres_analysis(
                    river_df, crossings, slopes
                )

                # summarise the results
                n_total_crossings = len(crossings)
                n_analysed = len(stat_df)
                n_raw_p = int((stat_df["p_value_raw"] < 0.05).sum()) if not stat_df.empty else 0
                n_fdr = int(stat_df.get("significant_fdr", pd.Series([], dtype=bool)).sum())
                n_truly = int(stat_df.get("truly_significant", pd.Series([], dtype=bool)).sum())
                med_abs_eff = float(np.median(np.abs(stat_df["raw_effect_mm_yr"]))) if not stat_df.empty else np.nan

                rows.append(
                    {
                        "window_km": win_km,
                        "effect_threshold_mm_yr": thr,
                        "distance_tolerance_m": tol_m,
                        "n_total_crossings": n_total_crossings,
                        "n_analysed": n_analysed,
                        "n_raw_p_lt_0_05": n_raw_p,
                        "n_fdr_sig": n_fdr,
                        "n_truly_sig": n_truly,
                        "median_abs_effect_mm_yr": med_abs_eff,
                        "background_r2": trend_data.get("r_value", 0.0) ** 2 if trend_data else np.nan,
                        "residual_std_mm_yr": trend_data.get("residual_std", np.nan) if trend_data else np.nan,
                    }
                )
            except Exception as e:
                log.exception("Sensitivity combo failed (win=%.2f, thr=%.2f, tol=%d): %s", win_km, thr, tol_m, e)
                rows.append(
                    {
                        "window_km": win_km,
                        "effect_threshold_mm_yr": thr,
                        "distance_tolerance_m": tol_m,
                        "n_total_crossings": np.nan,
                        "n_analysed": np.nan,
                        "n_raw_p_lt_0_05": np.nan,
                        "n_fdr_sig": np.nan,
                        "n_truly_sig": np.nan,
                        "median_abs_effect_mm_yr": np.nan,
                        "background_r2": np.nan,
                        "residual_std_mm_yr": np.nan,
                        "error": str(e),
                    }
                )

        # restore originals
        self.base_window_km = _orig["base_window_km"]
        self.effect_thr = _orig["effect_thr"]
        self.distance_tolerance_m = _orig["distance_tolerance_m"]

        sens_df = pd.DataFrame(rows).sort_values(["window_km", "effect_threshold_mm_yr", "distance_tolerance_m"])
        p = self.out_dir / f"{out_prefix}_sensitivity_grid.csv"
        sens_df.to_csv(p, index=False)
        log.info("Saved sensitivity grid CSV: %s", p.name)
        return sens_df


    # -----------------------------------------------------------------
    # 5.3  Run the whole pipeline
    # -----------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the complete analysis and return a dictionary with all artefacts."""
        try:
            # -------------------------------------------------
            # 0️)  Load data
            # -------------------------------------------------
            self._load_data()

            # -------------------------------------------------
            # 1️)  Pick the longest river and a distance column
            # -------------------------------------------------
            river_df = self._pick_longest_river()
            self._choose_distance_column(river_df)
            river_df = self._add_headwater_distance(river_df)
            if "LandCover_" in river_df.columns:
                lc_counts = (
                    river_df["LandCover_"].astype(str).str.lower().value_counts().head(12)
                )
                log.info("Top raw LandCover_ values: %s", lc_counts.to_dict())
            out_prefix = f"river_{river_df['MAIN_RIV'].iloc[0]}"


            # -------------------------------------------------
            # 2️)  Clean fault types (once for the whole dataset)
            # -------------------------------------------------
            self.faults["Fea_En_Clean"] = clean_fault_type(self.faults["Fea_En"])

            # -------------------------------------------------
            # 3️) Identify fault–river crossings
            # -------------------------------------------------
            crossings = self._identify_fault_crossings(river_df, self.faults)

            # -------------------------------------------------
            # 4️)  Compute (or reuse) channel slope & roughness
            # -------------------------------------------------
            slopes, roughness = self._compute_channel_metrics(
                river_df, distance_col="distance_from_headwater_km", elevation_col=self.elev_col
            )

            # -------------------------------------------------
            # 5️)  Aggregate data for nicer visualisation
            # -------------------------------------------------
            aggregated = aggregate_river_data_for_visualisation(
                river_df,
                distance_col="distance_from_headwater_km",
                velocity_col=self.vel_col,
                slopes=slopes,
                roughness=roughness,
                grid_spacing_km=0.5,
            )

            # -------------------------------------------------
            # 6️)  High‑resolution fault‑by‑fault analysis
            # -------------------------------------------------
            (
                statistical_results,
                effect_hist,
                trend_data,
                distances,
                velocities,
            ) = self._highres_analysis(river_df, crossings, slopes)

            # right after _highres_analysis(...)
            out_prefix = f"river_{river_df['MAIN_RIV'].iloc[0]}"


            # -------------------------------------------------
            # 7️)  Geomorphic correlation (slope / roughness)
            # -------------------------------------------------
            correlation_results, profiles_data = self._geomorphic_correlation_check(
                river_df,
                statistical_results,
                slopes,
                roughness,
                distance_col="distance_from_headwater_km",
                show_profiles=True,
            )

            # 7b) Permutation test – basin-level empirical null on raw+effect criteria
            n_perm = 200  
            obs_like = int(((statistical_results["p_value_raw"] < 0.05) &
                            (np.abs(statistical_results["raw_effect_mm_yr"]) > self.effect_thr)).sum())
            perm = self._permutation_test_overall(
                distances=trend_data["distances"],
                residuals=trend_data["velocity_residuals"],
                n_crossings=len(crossings),
                effect_thr=self.effect_thr,
                window_km=self.base_window_km,
                n_perm=n_perm,
            )
            self._write_permutation_summary(out_prefix, obs_like, perm, n_perm)

            # -------------------------------------------------
            # 8️) Build the figures (store them in `self.figures)
            # -------------------------------------------------
            fig1 = self._create_figure_1(trend_data, river_id=river_df["MAIN_RIV"].iloc[0], aggregated=aggregated)
            self._add_figure(fig1, "Basin 1 – Background trend", self.figures)

            fig2 = self._create_figure_2(
                trend_data,
                slopes,
                roughness,
                statistical_results,
                river_id=river_df["MAIN_RIV"].iloc[0],
                aggregated=aggregated,
            )
            self._add_figure(fig2, "Basin 1 – Multi‑panel longitudinal profiles", self.figures)

            fig3 = self._create_figure_3(effect_hist, statistical_results)
            self._add_figure(fig3, "Basin 1 – Effect‑size histogram & correction summary", self.figures)

            fig4 = self._create_figure_4(statistical_results, self.effect_thr)
            self._add_figure(fig4, "Basin 1 – Volcano plots (raw / FDR)", self.figures)

            fig5 = self._create_figure_5(
                river_df,
                distance_col="distance_from_headwater_km",
                statistical_results=statistical_results,
                basin_gdf=self.basin,
                faults_gdf=self.faults,                                  
                river_id=river_df["MAIN_RIV"].iloc[0],
            )

            if fig5:
                self._add_figure(fig5, "Basin 1 – Spatial context (significant faults)", self.figures)

            fig6 = self._create_figure_6(statistical_results)
            if fig6:
                self._add_figure(fig6, "Basin 1 – Drainage & land-cover summaries", self.figures)
    


            # -------------------------------------------------
            # 9️)  Export CSVs & summary table
            out_prefix = f"river_{river_df['MAIN_RIV'].iloc[0]}"
            statistical_results.to_csv(self.out_dir / f"{out_prefix}_fault_statistics.csv", index=False)
            correlation_results.to_csv(self.out_dir / f"{out_prefix}_geomorphic_correlation.csv", index=False)
            aggregated.to_csv(self.out_dir / f"{out_prefix}_aggregated_profile.csv", index=False)

            # 9b️  Drainage & land-cover context CSVs 
            for mask in ("truly", "fdr", "raw", "all"):
                try:
                    self._export_context_csvs(statistical_results, out_prefix, sig_mask=mask)
                except Exception as e:
                    log.exception("Context CSV export failed for mask %s: %s", mask, e)

            # 9c️⃣  Sensitivity grid (window, threshold, distance tolerance)
            try:
                self._run_sensitivity_grid(
                    river_df=river_df,
                    slopes=slopes,
                    roughness=roughness,
                    out_prefix=out_prefix,
                    window_km_list=(0.5, 1.0, 2.0),
                    effect_thresholds=(0.3, 0.5, 0.7),
                    distance_tolerances_m=(250, 500, 1000),
                )
            except Exception as e:
                log.exception("Sensitivity grid failed: %s", e)


            summary = {
                "river_id": river_df["MAIN_RIV"].iloc[0],
                "n_river_points": len(river_df),
                "river_length_km": distances.max() - distances.min(),
                "n_fault_crossings": len(crossings),
                "n_faults_analysed": len(statistical_results),
                "n_raw_significant": (statistical_results["p_value_raw"] < 0.05).sum(),
                "n_fdr_significant": statistical_results["significant_fdr"].sum(),
                "n_truly_significant": statistical_results["truly_significant"].sum(),
                "geomorphic_confounded_pct": (
                    correlation_results["any_geomorphic_change"].sum()
                    / len(correlation_results)
                    * 100
                    if len(correlation_results)
                    else np.nan
                ),
                "background_slope_mm_yr_per_km": trend_data["slope"],
                "background_r2": trend_data["r_value"] ** 2,
                "residual_std_mm_yr": trend_data["residual_std"],
            }
            pd.DataFrame([summary]).to_csv(self.out_dir / f"{out_prefix}_summary.csv", index=False)

            # -------------------------------------------------
            # 10️)  Return everything for downstream use
            # -------------------------------------------------
            log.info("Analysis complete – %d figures created.", len(self.figures))
            return {
                "figures": self.figures,
                "statistical_results": statistical_results,
                "correlation_results": correlation_results,
                "aggregated_data": aggregated,
                "profiles_data": profiles_data,
                "trend_data": trend_data,
                "effect_histogram": effect_hist,
                "summary_stats": summary,
                "river_data": river_df,
                "basin": self.basin,
                "selected_river_id": river_df["MAIN_RIV"].iloc[0],
            }

        except Exception as exc:
            log.exception("Fatal error during analysis: %s", exc)
            traceback.print_exc()
            raise

def stack_all_basin_outputs(out_dir: str = "output") -> Optional[pd.DataFrame]:
    """
    Scan out_dir for per-basin CSVs and write a MASTER table.
    Safe to call repeatedly; it just re-scans and overwrites MASTER_*.
    """
    import glob
    files = glob.glob(str(Path(out_dir) / "river_*_fault_statistics.csv"))
    if not files:
        log.info("No per-basin fault statistics to stack.")
        return None
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            stem = Path(f).stem 
            parts = stem.split("_")
            river_id = parts[1] if len(parts) > 1 else "unknown"
            df.insert(0, "river_id_filehint", river_id)
            frames.append(df)
        except Exception as e:
            log.warning("Skipping %s: %s", f, e)
    master = pd.concat(frames, ignore_index=True)
    out = Path(out_dir) / "MASTER_fault_statistics.csv"
    master.to_csv(out, index=False)
    log.info("Wrote %s with %d rows.", out.name, len(master))
    return master

# ------------------------------------------------------------
# 6️)  Convenience entry‑point (script mode)
# ------------------------------------------------------------

def main() -> None:
    """Run the pipeline with the hard‑coded example file locations."""
    basin_file = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\basin_10.shp"
    river_file = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\rivers\basin_10_rivers_elev_slp_rgh_vu_003_lc.shp"
    fault_file = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\faults\faults_single_parts.shp"

    analyser = RiverFaultAnalyser(
        basin_shp=basin_file,
        river_shp=river_file,
        fault_shp=fault_file,
        out_dir="output",
        base_window_km=1.0,
        effect_threshold_mm_yr=0.5,
        min_valid_points=20,
        distance_tolerance_m=500,
        background_resampling_km=0.5,
    )
    results = analyser.run()
    import os
    os.makedirs("output", exist_ok=True)

     # Save all figures in the results
    for idx, (title, fig) in enumerate(results["figures"], 1):
        safe_title = title.replace('/', '_').replace('\\', '_')
        fig_path = f"output/figure_{idx}_{safe_title}.png"
        fig.savefig(fig_path)
        print(f"Saved: {fig_path}")

    # -----------------------------------------------------------------
    # Final console summary for quick inspection
    # -----------------------------------------------------------------
    stats = results["statistical_results"]
    if not stats.empty:
        n_truly = stats["truly_significant"].sum()
        log.info("===> %d truly significant fault‑velocity jumps (FDR‑corrected).", n_truly)

    log.info("All artefacts written to the ‘output’ folder.")
    log.info("You can now import the figures programmatically, e.g.:\n")
    log.info("    fig, ax = results['figures'][0][1].axes")
    log.info("    fig.savefig('figure1.png')\n")


if __name__ == "__main__":
    main()

try:
    stack_all_basin_outputs(out_dir="output")
except Exception as e:
    log.exception("MASTER stack failed: %s", e)

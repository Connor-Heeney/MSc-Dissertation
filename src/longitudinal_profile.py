import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
import matplotlib.colors as mcolors

def _ensure_common_crs(basin, river, faults):
    print("\n CRS before conversion:")
    print(f"   basin : {basin.crs}")
    print(f"   river : {river.crs}")
    print(f"   faults: {faults.crs}")

    if basin.crs is None:
        raise ValueError("Basin layer has no CRS defined.")

    if basin.crs.is_geographic:
        centroid = basin.geometry.unary_union.centroid
        lon, lat = centroid.x, centroid.y
        zone = int((lon + 180)//6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        target_crs = f"EPSG:{epsg}"
        print(f"Reprojecting to {target_crs}")
        basin = basin.to_crs(target_crs)
    else:
        target_crs = basin.crs
        print(f"Basin already projected → {target_crs}")

    if river.crs != target_crs:
        river = river.to_crs(target_crs)
    if faults.crs != target_crs:
        faults = faults.to_crs(target_crs)

    print("\n CRS after conversion:")
    print(f"   basin : {basin.crs}")
    print(f"   river : {river.crs}")
    print(f"   faults: {faults.crs}")

    return basin, river, faults

def add_scale_bar(ax, bar_length_km=None, bar_height_frac=0.015, pad_frac_x=0.06, pad_frac_y=0.06):
    """
    Draw a 0–L scale bar that works for projected (meters) OR geographic (degrees) axes.
    If bar_length_km is None, choose a 'nice' 1–2–5 length from map width.
    """
    import numpy as np

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = x1 - x0
    dy = y1 - y0

    # Detect lon/lat axes
    in_degrees = (-180 <= x0 <= 180 and -180 <= x1 <= 180 and
                  -90 <= y0 <= 90 and -90 <= y1 <= 90)

    # Convert map width to kilometers
    if in_degrees:
        lat = 0.5 * (y0 + y1)
        km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(lat)))
        map_width_km = dx * km_per_deg_lon
    else:
        map_width_km = dx / 1000.0

    # Choose a nice bar length if not provided
    if bar_length_km is None:
        desired = max(1.0, 0.3 * map_width_km)
        nice = np.array([1,2,5,10,20,25,50,100,200,250,500,1000,2000])
        bar_length_km = float(nice[np.argmin(np.abs(nice - desired))])

    # Convert chosen length back to axis units (x-units)
    if in_degrees:
        bar_len_x = bar_length_km / km_per_deg_lon
    else:
        bar_len_x = bar_length_km * 1000.0

    # Keep the bar within half the map width
    if bar_len_x > 0.5 * dx:
        bar_len_x = 0.3 * dx
        bar_length_km = map_width_km * 0.3

        nice = np.array([1,2,5,10,20,25,50,100,200,250,500,1000,2000])
        bar_length_km = float(nice[np.argmin(np.abs(nice - bar_length_km))])

        if in_degrees:
            lat = 0.5 * (y0 + y1)
            km_per_deg_lon = max(1e-6, 111.32 * np.cos(np.deg2rad(lat)))
            bar_len_x = bar_length_km / km_per_deg_lon
        else:
            bar_len_x = bar_length_km * 1000.0

    # Pad and draw
    x_pad = x0 + pad_frac_x * dx
    y_pad = y0 + pad_frac_y * dy
    bar_height = bar_height_frac * dy

    ax.plot([x_pad, x_pad + bar_len_x], [y_pad, y_pad], color="k", lw=2, solid_capstyle='butt')
    ax.plot([x_pad, x_pad], [y_pad, y_pad + bar_height], color="k", lw=2)
    ax.plot([x_pad + bar_len_x, x_pad + bar_len_x], [y_pad, y_pad + bar_height], color="k", lw=2)

    fs = 11
    ax.text(x_pad, y_pad - 1.5 * bar_height, "0", ha="center", va="top", fontsize=fs)
    ax.text(x_pad + bar_len_x, y_pad - 1.5 * bar_height, f"{int(bar_length_km) if bar_length_km>=10 else bar_length_km:g}",
            ha="center", va="top", fontsize=fs)
    ax.text(x_pad + bar_len_x/2, y_pad - 2.6 * bar_height, "km", ha="center", va="top", fontsize=fs, fontweight="bold")



def add_north_arrow(ax, pad=0.05, size=0.1):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx, dy = x1 - x0, y1 - y0
    xc = x0 + pad*dx
    yc = y1 - pad*dy
    arrow_len = dy * size
    ax.annotate("N",
                xy=(xc, yc + arrow_len/2),
                xytext=(xc, yc - arrow_len/2),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"),
                ha="center", va="center",
                fontsize=14, fontweight="bold")

def create_figure_6_basin_overview(river_points, basin, selected_river_id):
    print("\n Figure 6: Basin River Network Overview")
    dist_cols = ["distance", "DIST_DN_KM", "DIST_UP_KM", "DIST_UP_M"]
    dc = next((c for c in dist_cols if c in river_points.columns), None)
    if dc:
        river_points["distance_m"] = (
            river_points[dc].astype(float)
            .pipe(lambda s: np.where(s.max()<500, s*1000, s))
        )
    else:
        river_points["distance_m"] = np.arange(len(river_points))

    stats = []
    for riv, grp in river_points.groupby("MAIN_RIV"):
        stats.append({
            "MAIN_RIV": riv,
            "points": len(grp),
            "length_km": (grp["distance_m"].max() - grp["distance_m"].min())/1000
        })
    stats_df = pd.DataFrame(stats).sort_values("length_km", ascending=False)
    total_pts = len(river_points)
    sel_len = stats_df.loc[stats_df["MAIN_RIV"]==selected_river_id, "length_km"].iloc[0]
    print(f"   rivers: {len(stats_df)}, points: {total_pts:,}")
    print(f"   selected {selected_river_id}: {sel_len:.1f} km")

    fig, ax = plt.subplots(figsize=(16, 8))
    basin.plot(ax=ax, facecolor="lightblue", edgecolor="navy", linewidth=2, alpha=0.1)

    others = river_points[river_points["MAIN_RIV"]!=selected_river_id]
    if len(others):
        step = max(1, len(others)//5000)
        others.iloc[::step].plot(ax=ax, color="lightgray", markersize=0.5, alpha=0.6)

    sel = river_points[river_points["MAIN_RIV"]==selected_river_id]
    cmap = "terrain"
    if not sel.empty and "ELEV_1" in sel.columns:
        sc = ax.scatter(sel.geometry.x, sel.geometry.y,
                        c=sel["ELEV_1"], cmap=cmap,
                        s=8, edgecolors="none", zorder=5)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, aspect=20)
        cb.set_label("Elevation (m)", fontweight="bold")

    # --- Robust head/mouth using whichever distance has real variation ---
    def _pick_head_mouth(sel_df):
        cand_specs = [
            ("DIST_UP_KM", lambda v: v.astype(float)),                         # from head (km)
            ("DIST_UP_M",  lambda v: v.astype(float)/1000.0),                  # from head (m→km)
            ("DIST_DN_KM", lambda v: (v.astype(float).max() - v.astype(float))),  # to outlet → from head
            ("distance_m", lambda v: v.astype(float)/1000.0),                  # meters → km
            ("distance",   None),                                              # ambiguous; treat below
        ]

        candidates = []
        for name, conv in cand_specs:
            if name in sel_df.columns:
                vals = pd.to_numeric(sel_df[name], errors="coerce")
                if name == "distance":
                    # Heuristic: meters if big, else km
                    vals = (vals/1000.0) if np.nanmax(vals) > 2000 else vals
                elif conv is not None:
                    vals = conv(vals)

                vals = vals.to_numpy()
                if np.all(np.isnan(vals)):
                    continue

                # normalise to start at 0 from headwater
                vals = vals - np.nanmin(vals)
                rng = float(np.nanmax(vals) - np.nanmin(vals))
                candidates.append((name, vals, rng))

        if candidates:
            # pick the field with the **largest range**
            best_name, best_vals, best_rng = max(candidates, key=lambda t: t[2])
            sel_df = sel_df.assign(dist_from_head_km=best_vals)
            i_head = int(np.nanargmin(best_vals))
            i_mouth = int(np.nanargmax(best_vals))
            head = sel_df.iloc[i_head]
            mouth = sel_df.iloc[i_mouth]
            # if ties collapse to same point, fall back to elevation extremes
            if head.geometry.equals(mouth.geometry) and "ELEV_1" in sel_df.columns:
                head = sel_df.loc[sel_df["ELEV_1"].idxmax()]
                mouth = sel_df.loc[sel_df["ELEV_1"].idxmin()]
            print(f"   Head/Mouth by {best_name} (range ≈ {best_rng:.2f} km)")
            return head, mouth

        # this is the last resort: elevation extremes
        if "ELEV_1" in sel_df.columns:
            print("   No usable distance field; using elevation extremes.")
            return sel_df.loc[sel_df["ELEV_1"].idxmax()], sel_df.loc[sel_df["ELEV_1"].idxmin()]
        else:
            raise KeyError("No usable distance or elevation fields to determine head/mouth.")

    # use it on the selected river points
    head, mouth = _pick_head_mouth(sel)


    ax.scatter(head.geometry.x, head.geometry.y,
               marker="^", s=200, color="darkgreen",
               edgecolor="white", linewidth=2.5, zorder=10)
    ax.scatter(mouth.geometry.x, mouth.geometry.y,
               marker="v", s=200, color="darkblue",
               edgecolor="white", linewidth=2.5, zorder=10)

    xmin, ymin, xmax, ymax = basin.total_bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    norm = mcolors.Normalize(vmin=sel["ELEV_1"].min(), vmax=sel["ELEV_1"].max()) if not sel.empty else None
    cmap_obj = plt.colormaps[cmap]
    col_val = cmap_obj(0.7) if not sel.empty else "red"
    legend_items = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="lightgray",
               markersize=6, label="Other rivers", linestyle="None", alpha=0.6),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=col_val,
               markersize=8, label=f"River {selected_river_id}", linestyle="None"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor="darkgreen",
               markersize=12, label="Headwater", linestyle="None", markeredgecolor="white", markeredgewidth=2),
        Line2D([0],[0], marker="v", color="w", markerfacecolor="darkblue",
               markersize=12, label="Mouth", linestyle="None", markeredgecolor="white", markeredgewidth=2),
        Patch(facecolor="lightblue", edgecolor="navy", alpha=0.3,
              linewidth=2, label="Basin boundary"),
    ]
    ax.legend(handles=legend_items, loc="upper right",
              frameon=True, fancybox=True, shadow=True, fontsize=11)

  # Let the bar auto-size to map width & units
    add_scale_bar(ax, bar_length_km=None, pad_frac_y=0.10)
    add_north_arrow(ax)

    ax.set_title(
        f"Basin 14 River Network Overview\n"
        f"Selected River {selected_river_id} for Fault Analysis",
        fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_aspect("equal")
    plt.tight_layout()
    return fig, stats_df

def create_figure_7_longitudinal_profiles(
        river_points, faults, selected_river_id, significance_threshold=0.5):
    """
    Figure 7: Longitudinal profiles (Elevation + Vu) with mapped fault intersections.
    - Robust to different distance fields (distance, DIST_DN_KM, DIST_UP_KM, DIST_UP_M, distance_m)
    - Distances are plotted in km (auto-converted if meters)
    - 'significant' faults are those where a local mean upstream–downstream ΔVu exceeds threshold
    """
    print("Creating Longitudinal Profiles")

    # ---- pick a distance column and make km ----
    rp = river_points.loc[river_points["MAIN_RIV"] == selected_river_id].copy()
    if rp.empty:
        raise ValueError(f"No points found for MAIN_RIV={selected_river_id}")

    # prefer an existing 'distance_m' if it’s already in the data
    dist_cols = ["distance_m", "distance", "DIST_DN_KM", "DIST_UP_KM", "DIST_UP_M"]
    dc = next((c for c in dist_cols if c in rp.columns), None)
    if dc is None:
        raise KeyError("No distance column found among: distance_m, distance, DIST_DN_KM, DIST_UP_KM, DIST_UP_M")

    # build meters from whatever we have (treat values <500 as km, else meters)
    vals = pd.to_numeric(rp[dc], errors="coerce").values
    if np.nanmax(vals) < 500:       
        dist_m = vals * 1000.0
    else:                           
        dist_m = vals.copy()

    # headwater-referenced km (start at 0), sorted downstream
    rp["distance_km"] = dist_m / 1000.0
    rp["distance_km"] -= np.nanmin(rp["distance_km"])
    rp = rp.sort_values("distance_km").reset_index(drop=True)
    total_length = float(np.nanmax(rp["distance_km"]))
    print(f"   Using distance field '{dc}' → 0–{total_length:.1f} km")


    print(f"   River length: {total_length:.1f} km | Profile points: {len(rp)}")

    # ---- find nearby faults (within tolerance) and estimate local ΔVu ----
    tolerance_km = 0.5  
    fault_rows = []
    for _, ft in faults.iterrows():
        try:
            d = rp.geometry.distance(ft.geometry)  # CRS already projected by _ensure_common_crs
            min_km = float(d.min()) / 1000.0
            if min_km <= tolerance_km:
                idx = int(d.idxmin())
                # local upstream/downstream mean windows (in indices)
                halfwin = 20
                i0 = max(0, idx - halfwin)
                i1 = min(len(rp), idx + halfwin)
                if (i1 - i0) > 10 and "VEL_1" in rp.columns:
                    up = float(rp.iloc[i0:idx]["VEL_1"].mean()) if i0 < idx else np.nan
                    dn = float(rp.iloc[idx:i1]["VEL_1"].mean()) if idx < i1 else np.nan
                    dv = np.nan if (np.isnan(up) or np.isnan(dn)) else abs(dn - up)
                else:
                    dv = np.nan

                # map a readable fault type if present
                ftype = "Unknown"
                for cand in ["Fea_En", "FEA_EN", "type", "TYPE", "Class", "CLASS", "SlipSense"]:
                    if cand in ft.index and pd.notnull(ft[cand]):
                        raw = str(ft[cand]).lower()
                        if "reverse" in raw:
                            ftype = "Reverse"
                        elif "normal" in raw:
                            ftype = "Normal"
                        elif "left" in raw:
                            ftype = "Left Lateral"
                        elif "right" in raw:
                            ftype = "Right Lateral"
                        break

                row = rp.iloc[idx]
                fault_rows.append({
                    "distance_km": float(row["distance_km"]),
                    "elevation": float(row["ELEV_1"]) if "ELEV_1" in rp.columns else np.nan,
                    "velocity": float(row["VEL_1"]) if "VEL_1" in rp.columns else np.nan,
                    "fault_type": ftype,
                    "velocity_jump": dv
                })
        except Exception:
            continue

    fault_df = pd.DataFrame(fault_rows)
    if not fault_df.empty:
        sig_faults = fault_df[fault_df["velocity_jump"] > float(significance_threshold)].copy()
        print(f"   Fault intersections: {len(fault_df)} | Significant (|Δv|>{significance_threshold}): {len(sig_faults)}")
    else:
        sig_faults = pd.DataFrame()
        print("   No fault intersections found within tolerance.")

    # ---- Figure with stacked panels (Elevation top, Vu bottom) ----
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.05)

    # PANEL 1: elevation
    ax1 = fig.add_subplot(gs[0])
    if "ELEV_1" in rp.columns:
        ax1.scatter(rp["distance_km"], rp["ELEV_1"], c="darkblue", s=3, alpha=0.6, edgecolors="none")
        if len(rp) > 50:
            # Savitzky–Golay smooth
            wlen = min(51, (len(rp)//10)*2 + 1)  # odd
            ax1.plot(rp["distance_km"], savgol_filter(rp["ELEV_1"], wlen, 3),
                     "navy", lw=2, alpha=0.8, label="Smoothed profile")
    ax1.set_ylabel("Elevation (m)", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Basin 14 Longitudinal River Profiles with Fault Intersections\n"
        f"River {selected_river_id}", fontsize=16, fontweight="bold", pad=20
    )
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.tick_params(axis='x', which='both', bottom=False, top=False,
                    labelbottom=False, labeltop=False)
    ax1.set_xlabel("")  # this is to ensure no accidental x-label on ax1

    # PANEL 2: velocity
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if "VEL_1" in rp.columns:
        pos = rp["VEL_1"] > 0
        ax2.scatter(rp.loc[pos, "distance_km"], rp.loc[pos, "VEL_1"], c="darkred",
                    s=3, alpha=0.6, label="Uplift", edgecolors="none")
        ax2.scatter(rp.loc[~pos, "distance_km"], rp.loc[~pos, "VEL_1"], c="darkblue",
                    s=3, alpha=0.6, label="Subsidence", edgecolors="none")
        if len(rp) > 50:
            ax2.plot(rp["distance_km"], savgol_filter(rp["VEL_1"], wlen, 3),
                     "darkgreen", lw=2, alpha=0.8, label="Smoothed velocity")
    ax2.axhline(0, color="black", lw=1, alpha=0.5)
    ax2.set_xlabel("Distance from Headwater (km)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Vertical Velocity (mm/yr)", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.legend(loc="upper right", fontsize=10)

    # vertical markers at significant faults
    colors = {"Reverse": "crimson", "Normal": "blue",
              "Left Lateral": "green", "Right Lateral": "purple", "Unknown": "gray"}
    if not sig_faults.empty:
        seen = set()
        for _, r in sig_faults.iterrows():
            col = colors.get(r["fault_type"], "gray")
            ax1.axvline(r["distance_km"], color=col, ls="--", lw=1.5, alpha=0.7)
            ax2.axvline(r["distance_km"], color=col, ls="--", lw=1.5, alpha=0.7)
            seen.add(r["fault_type"])
        # legend for faults on top panel
        fault_handles = [Line2D([0],[0], color=colors.get(t,"gray"), ls="--", lw=2, label=t)
                         for t in sorted(seen)]
        if fault_handles:
            leg = ax1.legend(handles=fault_handles, loc="upper right", fontsize=10, title="Fault Types")
            leg.get_title().set_fontsize(11); leg.get_title().set_fontweight("bold")

    # x-limits and ticks
    ax1.set_xlim(0, total_length); ax2.set_xlim(0, total_length)
    tick_spacing = 10
    ax2.set_xticks(np.arange(0, np.ceil(total_length)+1, tick_spacing))
    from matplotlib.ticker import FormatStrFormatter, MultipleLocator
    ax2.xaxis.set_major_locator(MultipleLocator(tick_spacing))   # lock spacing
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%d'))      # force integers
    ax2.tick_params(axis='x', which='major', labelsize=12)       # ensure visible
    ax2.set_xticklabels([f"{int(t)}" for t in ax2.get_xticks()]) # explicitly draw


    plt.tight_layout()

    # quick profile stats
    if "ELEV_1" in rp.columns:
        elev_drop = float(rp["ELEV_1"].max() - rp["ELEV_1"].min())
        grad = elev_drop / max(total_length, 1e-6)
        print(f"\n   PROFILE STATS: Δelev={elev_drop:.0f} m ({grad:.1f} m/km), "
              f"Vu [{rp['VEL_1'].min():.1f}, {rp['VEL_1'].max():.1f}] mm/yr, "
              f"mean {rp['VEL_1'].mean():.2f} ± {rp['VEL_1'].std():.2f}")
    return fig, sig_faults


def create_all_basin_figures(basin_file, river_file, fault_file):
    basin = gpd.read_file(basin_file)
    rivers = gpd.read_file(river_file)
    faults = gpd.read_file(fault_file)
    basin, rivers, faults = _ensure_common_crs(basin, rivers, faults)
    faults = gpd.clip(faults, basin)
    length_cols = ["DIST_DN_KM","distance","DIST_UP_KM"]
    lc = next((c for c in length_cols if c in rivers.columns), None)
    if lc is None:
        raise KeyError("No length column found.")
    lengths = rivers.groupby("MAIN_RIV")[lc].agg(["min","max"])
    lengths["length"] = lengths["max"]-lengths["min"]
    selected = lengths["length"].idxmax()
    print(f"\n Selected River: {selected}")
    fig6, stats = create_figure_6_basin_overview(rivers, basin, selected)
    fig7, significant_faults = create_figure_7_longitudinal_profiles(rivers, faults, selected, significance_threshold=0.5)
    plt.show()
    return fig6, stats

if __name__ == "__main__":
    basin_fp = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\basin_6.shp"
    river_fp = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\rivers\basin_6_rivers_elev_slp_rgh_vu_003_lc.shp"
    fault_fp = r"C:\Users\connor.heeney\OneDrive - ESA\Documents\Personal\Dissertation\core\data\faults\faults_single_parts.shp"
    create_all_basin_figures(basin_fp, river_fp, fault_fp)

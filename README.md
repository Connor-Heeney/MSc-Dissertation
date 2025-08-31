# MSc Dissertation  

This repository contains the code and outputs for my MSc dissertation on **vertical land motion and fault–river interactions in the northeastern Tibetan Plateau**, using InSAR-derived velocities and GNSS benchmarking.  

## Repository Structure  

```plaintext
├── src/                      # Source code
│   ├── descriptive_stats/    # Scripts for descriptive statistics
│   ├── gnss_benchmarking/    # Scripts for GNSS vs InSAR benchmarking
│   ├── longitudinal_profile_overview/         # River profile analysis
│   ├── longitudinal_profile_advanced_analytics/  # Step-detection, clustering, etc.
│   ├── descriptive_stats.py
│   ├── gnss_benchmarking.py
│   ├── longitudinal_profile.py
│   ├── longitudinal_profile_advanced_analytics.py
│   ├── test_*                # Validation and testing scripts
│
├── outputs/                  # Model and analysis outputs
│   ├── Basin_1/              # Outputs for individual basins
│   ├── Basin_2/  
│   ├── …  
│   └── Basin_14/  
│   └── gnss_benchmarking/    # Global benchmarking results (not per-basin)
│
└── README.md                 # Project overview

```

# Usage

Clone the repository:

git clone https://github.com/Connor-Heeney/MSc-Dissertation.git
cd MSc-Dissertation


# Notes

All scripts are written in Python 3.10.11

Outputs are generated directly by the pipeline; rerunning scripts will overwrite existing results.

Dataset references: Vu dataset (Ou et al., 2022), HydroSHEDS (https://www.hydrosheds.org/), CAFD faults (https://essd.copernicus.org/articles/16/3391/2024/), Esri Land Cover (https://livingatlas.arcgis.com/landcover/), GNSS (https://data.mendeley.com/datasets/k6t8rkh4pd/1), Copernicus DEM GLO-30 (https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3).

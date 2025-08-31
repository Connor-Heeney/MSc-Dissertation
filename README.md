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
├── outputs/                  # Model and analysis outputs within each subfolder
│   ├── descriptive_stats/              
│   ├── gnss_benchmarking/    # Global benchmarking results (not per-basin)
│   ├── longitudinal_profile_overview
│   └── longitudinal_profile_advanced_analytics
│
│├── data/
│   ├── rivers/               # rivers for each basin with sampled Vu, elevation, roughness, land cover, and slope              
│   ├── Basin_1.shp         
│   ├── Basin_2.shp
│   ├── …  
│   └── Basin_14.shp
└── README.md                 # Project overview

```

# Usage

Clone the repository:

git clone https://github.com/Connor-Heeney/MSc-Dissertation.git
cd MSc-Dissertation


# Notes

All scripts are written in Python 3.10.11

Outputs are generated directly by the pipeline; rerunning scripts will overwrite existing results.

The file sizes of the fault and GNSS data are too large to upload to this repository. To download both layers, please follow the links provided below. The GNSS data is ready to use as soon as it is downloaded, whilst the fault data needs to be loaded into QGIS and converted from singlepart to multipart before processing. 

Dataset references: 

- Vu dataset (Ou et al., 2022 - https://data.ceda.ac.uk/neodc/comet/publications_data/Ou_et_al_JGR_2022/v1.0/5_cartesian_velocities)
- HydroSHEDS (https://www.hydrosheds.org/)
- CAFD faults (https://essd.copernicus.org/articles/16/3391/2024/)
- Esri Land Cover (https://livingatlas.arcgis.com/landcover/)
- GNSS (1970 - 2017) (https://data.mendeley.com/datasets/k6t8rkh4pd/1)
- Copernicus DEM GLO-30 (https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3)

### Setup guide

1. Setup env with `hydrolib` package (if not done)
```
conda create -n delft3d_dhydamo python=3.12 -y
conda activate delft3d_dhydamo
conda install -c conda-forge notebook -y
pip install hydrolib
```

2. Import template dataset in `template/`
3. Change directory dataset in notebook

### Houston dataset

1. Download raw dataset from Drive
2. Put the dataset in folder `houston/` like this
```
houston/raw/
  ├── dataset.docx                      # mô tả dữ liệu
  ├── dem1m/                            # 24 tile DEM 1 m (USGS, Houston B24)
  │   └── USGS_1M_15_x{23..28}y{329..332}_TX_Houston_B24.tif
  ├── flood_landuse/                    # thư mục rỗng
  ├── FloodManagement.gpkg
  ├── LandbaseAndRoads.gpkg
  ├── Stormdrain.gpkg
  ├── Topography.gpkg
  ├── WastewaterUtilities.gpkg
  └── WaterUtilities_noSysValve.gpkg
```
3. Do data processing with scripts in `houston/` or in `tool/scripts/`
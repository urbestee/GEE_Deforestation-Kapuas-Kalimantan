# Monitoring Deforestation in Kapuas, Central Kalimantan using Google Earth Engine

This project leverages Google Earth Engine (GEE) to monitor deforestation trends in Kapuas Regency, Central Kalimantan, Indonesia from 2015 to 2023 using multispectral satellite imagery, spectral indices, machine learning classification (Random Forest), and hotspot analysis (VIIRS).

# Project Goals

- Detect land cover changes between 2015 and 2023  
- Analyze fire-affected deforestation areas using VIIRS hotspot data  
- Identify cleared land (open areas) potentially suitable for oil palm expansion  
- Evaluate classification performance and mapping accuracy

# Tools & Technologies

- Google Earth Engine (GEE)
- Sentinel-2 SR (2023)
- Landsat 8 SR (2015)
- VIIRS VNP14A1 Fire Product
- Random Forest Classifier
- Spectral Indices: NDVI, NBR, NDWI

# Study Area

Location: Kapuas Regency, Central Kalimantan, Indonesia  
Boundary Source: FAO GAUL Admin Level 2

# Methodology

#1. Preprocessing
- Cloud masking for Sentinel-2 and Landsat 8
- Clipping to AOI (Kapuas)
- Calculation of vegetation indices:
  - `NDVI = (NIR - Red) / (NIR + Red)`
  - `NBR  = (NIR - SWIR2) / (NIR + SWIR2)`
  - `NDWI = (Green - NIR) / (Green + NIR)`

#2. Classification
- Manually digitized training samples (Forest, Non-Forest, Water)
- Random Forest classifier with 50 trees
- Applied on stacked bands + indices
- Classification for years: 2015 and 2023

#3. Accuracy Assessment
- 70% training, 30% validation
- Confusion matrix + Over

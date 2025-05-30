// ======================================================
// Remote Sensing Analysis in Kalimantan using GEE
// Objective: Detect deforestation, classify land cover, analyze fire hotspots
// Tools: Sentinel-2, VIIRS, Landsat 8, Random Forest, NDVI, NBR, NDWI
// Author: Bella Esti Ajeng Syahputri
// Platform: Google Earth Engine JavaScript API
// ======================================================

// 1. Area of Interest (AOI) - Kapuas, Central Kalimantan
var kalimantan = ee.FeatureCollection('FAO/GAUL/2015/level2')
  .filter(ee.Filter.eq('ADM2_NAME', 'Kapuas'))
  .filter(ee.Filter.eq('ADM1_NAME', 'Kalimantan Tengah'));
Map.centerObject(kalimantan, 8);
Map.addLayer(kalimantan, {}, 'Kapuas AOI');

// 2. Sentinel-2 Preprocessing (Year: 2023)
function maskS2clouds(image) {
  var scl = image.select('SCL');
  return image.updateMask(scl.neq(3)).updateMask(scl.neq(9));
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR')
  .filterBounds(kalimantan)
  .filterDate('2023-01-01', '2023-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
  .map(maskS2clouds)
  .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
  .median()
  .clip(kalimantan);
Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Sentinel-2 RGB 2023');

// 3. Vegetation Indices (NDVI, NBR, and NDWI) extracted from Sentinel-2 imagery
var ndvi2023 = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
var nbr2023 = s2.normalizedDifference(['B8', 'B12']).rename('NBR');
var ndwi2023 = s2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var indices2023 = s2.addBands([ndvi2023, nbr2023, ndwi2023]);

// 4. Landsat 8 Preprocessing (Year: 2015)
function maskL8sr(image) {
  var qa = image.select('QA_PIXEL');
  var cloud = 1 << 3;
  var cloudShadow = 1 << 4;
  var snow = 1 << 5;
  var mask = qa.bitwiseAnd(cloud).eq(0)
    .and(qa.bitwiseAnd(cloudShadow).eq(0))
    .and(qa.bitwiseAnd(snow).eq(0));
  return image.updateMask(mask);
}

var l8_raw = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(kalimantan)
  .filterDate('2015-01-01', '2015-12-31')
  .filter(ee.Filter.lt('CLOUD_COVER', 30))
  .map(maskL8sr)
  .median()
  .clip(kalimantan);

var l8 = l8_raw.multiply(0.0000275).add(-0.2)
  .select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
          ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']);
Map.addLayer(l8, {bands: ['B4', 'B3', 'B2'], min: 0, max: 0.3}, 'Landsat 8 RGB 2015');

// 4.1 Vegetation Indices extracted from Landsat 8 imagery
var ndvi2015 = l8.normalizedDifference(['B5', 'B4']).rename('NDVI');
var nbr2015 = l8.normalizedDifference(['B5', 'B7']).rename('NBR');
var ndwi2015 = l8.normalizedDifference(['B3', 'B5']).rename('NDWI');
var indices2015 = l8.addBands([ndvi2015, nbr2015, ndwi2015]);

// 5. VIIRS Fire Hotspots (2023)
var viirs = ee.ImageCollection('NOAA/VIIRS/001/VNP14A1')
  .filterBounds(kalimantan)
  .filterDate('2023-01-01', '2023-12-31');

var fireMask = viirs.select('MaxFRP').mean().clip(kalimantan);
Map.addLayer(fireMask.updateMask(fireMask.gt(0)), {
  min: 10, max: 100, palette: ['yellow', 'orange', 'red']
}, 'VIIRS MaxFRP 2023');

// 6. Training Samples (3 Classes: Forest, Non-Forest, Water)
var forestPoints = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point(114.3202, -1.4811), {'landcover': 0}),
  ee.Feature(ee.Geometry.Point(114.3627, -1.6031), {'landcover': 0}),
  ee.Feature(ee.Geometry.Point(114.1025, -1.8124), {'landcover': 0}),
  ee.Feature(ee.Geometry.Point(114.6143, -2.6542), {'landcover': 0}),
]);

var nonForestPoints = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point(114.2921, -1.8069), {'landcover': 1}),
  ee.Feature(ee.Geometry.Point(114.1983, -1.1586), {'landcover': 1}),
  ee.Feature(ee.Geometry.Point(114.3659, -1.3505), {'landcover': 1}),
  ee.Feature(ee.Geometry.Point(114.3658, -1.3478), {'landcover': 1})
]);

var waterPoints = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Point(114.1730, -1.1285), {'landcover': 2}),
  ee.Feature(ee.Geometry.Point(114.2902, -1.2798), {'landcover': 2}),
  ee.Feature(ee.Geometry.Point(114.3888, -1.3844), {'landcover': 2}),
  ee.Feature(ee.Geometry.Point(114.4691, -2.4308), {'landcover': 2})
]);

var trainingSamples = forestPoints.merge(nonForestPoints).merge(waterPoints);
Map.addLayer(trainingSamples, {color: 'red'}, 'Training Samples');

// 7. Land Cover Classification for 2023 using Random Forest
var bands23 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NBR', 'NDWI'];
var training23 = indices2023.select(bands23).sampleRegions({
  collection: trainingSamples,
  properties: ['landcover'],
  scale: 10
});
var classifier23 = ee.Classifier.smileRandomForest(50).train({
  features: training23,
  classProperty: 'landcover',
  inputProperties: bands23
});
var classified23 = indices2023.select(bands23).classify(classifier23);
Map.addLayer(classified23, {min: 0, max: 2, palette: ['green', 'yellow', 'blue'], opacity: 0.6}, 'Land Cover 2023');

// 8. Land Cover Classification for 2015 using Random Forest
var bands15 = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'NDVI', 'NBR', 'NDWI'];
var training15 = indices2015.select(bands15).sampleRegions({
  collection: trainingSamples,
  properties: ['landcover'],
  scale: 30
});
var classifier15 = ee.Classifier.smileRandomForest(50).train({
  features: training15,
  classProperty: 'landcover',
  inputProperties: bands15
});
var classified15 = indices2015.select(bands15).classify(classifier15);
Map.addLayer(classified15, {min: 0, max: 2, palette: ['green', 'yellow', 'blue'], opacity: 0.6}, 'Land Cover 2015');

// 9. Validation of Classification Accuracy (2023)

var validation23 = training23.randomColumn();
var trainSet23 = validation23.filter(ee.Filter.lt('random', 0.7));
var testSet23 = validation23.filter(ee.Filter.gte('random', 0.7));

var trainedClassifier23 = ee.Classifier.smileRandomForest(50).train({
  features: trainSet23,
  classProperty: 'landcover',
  inputProperties: bands23
});
var validated23 = testSet23.classify(trainedClassifier23);
var confusionMatrix23 = validated23.errorMatrix('landcover', 'classification');
print('Confusion Matrix 2023:', confusionMatrix23);
print('Accuracy 2023:', confusionMatrix23.accuracy());

// 10. Validation of Classification Accuracy (2015)

var validation15 = training15.randomColumn();
var trainSet15 = validation15.filter(ee.Filter.lt('random', 0.7));
var testSet15 = validation15.filter(ee.Filter.gte('random', 0.7));

var trainedClassifier15 = ee.Classifier.smileRandomForest(50).train({
  features: trainSet15,
  classProperty: 'landcover',
  inputProperties: bands15
});
var validated15 = testSet15.classify(trainedClassifier15);
var confusionMatrix15 = validated15.errorMatrix('landcover', 'classification');
print('Confusion Matrix 2015:', confusionMatrix15);
print('Accuracy 2015:', confusionMatrix15.accuracy());

// 11. Change Detection Analysis of Land Cover (2015 to 2023)
var changeMap = classified23.subtract(classified15).rename('Change');
Map.addLayer(changeMap, {min: -2, max: 2, palette: ['red', 'white', 'blue']}, 'Change Detection');

// 12. Spatial Overlay: Deforestation and Burned Areas
var deforestToNonForest = classified15.eq(0).and(classified23.eq(1));
var burnedArea = fireMask.gt(10);
var deforestAndBurned = deforestToNonForest.and(burnedArea);
Map.addLayer(deforestAndBurned.updateMask(deforestAndBurned), {
  palette: ['orange']
}, 'Deforestation → Burned Area');

// 13. Detection of Open Areas after Deforestation (Oil Palm Suitability)
var ndviThresh = 0.5;
var nbrThresh = 0.1;
var openLand = ndvi2023.lt(ndviThresh).and(nbr2023.lt(nbrThresh));
var openAfterForest = deforestToNonForest.and(openLand);
Map.addLayer(openAfterForest.updateMask(openAfterForest), {
  palette: ['yellow']
}, 'Open Land After Deforestation');

// 14. Multi-Criteria Overlay: Deforestation, Fire, and Open Areas
var potentialPalmExpansion = deforestToNonForest.and(burnedArea).and(openLand);
Map.addLayer(potentialPalmExpansion.updateMask(potentialPalmExpansion), {
  palette: ['red']
}, 'Potential Palm Oil Expansion');

// 15. Chart: Land Cover Change Statistics
// Merge the 2015 and 2023 classification maps
var combined = classified15.multiply(10).add(classified23).rename('transition');

// Calculate the transition histogram
var transitionStats = combined.reduceRegion({
  reducer: ee.Reducer.frequencyHistogram(),
  geometry: kalimantan.geometry(),
  scale: 30,
  maxPixels: 1e13,
  bestEffort: true
});

// Convert the histogram into a dictionary
var transitionDict = ee.Dictionary(transitionStats.get('transition'));

// Calculate the total number of pixels
var totalPixels = transitionDict.values().reduce(ee.Reducer.sum());

// Generate a percentage-based chart
var keys = transitionDict.keys().sort();

var percentages = keys.map(function(k) {
  var count = ee.Number(transitionDict.get(k));
  return count.divide(totalPixels).multiply(100); // % Change Percentage
});

var labels = keys.map(function(k) {
  var code = ee.Number.parse(k);
  var fromTo = code.divide(10).floor().format('%d').cat('→').cat(code.mod(10).format('%d'));
  return fromTo;
});

// Display the percentage chart
var chart = ui.Chart.array.values({
  array: ee.Array(percentages),
  axis: 0,
  xLabels: labels
}).setChartType('ColumnChart')
  .setOptions({
    title: 'Land Cover Transition Percentage (2015 → 2023)',
    hAxis: {title: 'Transition (From → To)', slantedText: true},
    vAxis: {
      title: 'Percentage (%)',
      format: '#,##0.0'
    },
    colors: ['#d95f02']
  });

print(chart);

// 16. Area Statistics (in Hectares) for Each Condition
// Function to Calculate Area (in Hectares)
function calculateArea(mask, name) {
  var areaImage = mask.multiply(ee.Image.pixelArea()).divide(10000); // Convert to hectares
  var stats = areaImage.reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: kalimantan.geometry(),
    scale: 30,
    maxPixels: 1e13
  });
  print('Area size (' + name + ') in hectares:', stats);
}

// Compute Area for Each Condition
calculateArea(deforestToNonForest, 'Deforestation (Forest → Non-Forest)');
calculateArea(burnedArea, 'Burned Area');
calculateArea(openAfterForest, 'Open Land Post-Deforestation');
calculateArea(potentialPalmExpansion, 'High-Risk Zone for Oil Palm Expansion');

// ==================
// Advanced Analysis
// ==================

//Image Segmentation Using SNIC (2023)
var snic2023 = ee.Algorithms.Image.Segmentation.SNIC({
  image: indices2023.select(['NDVI', 'NBR', 'NDWI']),
  size: 5,
  compactness: 1,
  connectivity: 8,
  neighborhoodSize: 128,
  seeds: ee.Algorithms.Image.Segmentation.seedGrid(10)
});
Map.addLayer(snic2023.select('clusters'), {}, 'SNIC Segmentation 2023');

// Image Segmentation Using SNIC (2015)
var snic2015 = ee.Algorithms.Image.Segmentation.SNIC({
  image: indices2015.select(['NDVI', 'NBR', 'NDWI']),
  size: 5,
  compactness: 1,
  connectivity: 8,
  neighborhoodSize: 128,
  seeds: ee.Algorithms.Image.Segmentation.seedGrid(10)
});
Map.addLayer(snic2015.select('clusters'), {}, 'SNIC Segmentation 2015');


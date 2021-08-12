require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../utils/load-csv');
const tf_knn = require('./tf_knn');

const useSmallData = false;
const testSize = 10;
const k = 10;
const csvFile = useSmallData ? 'kc_house_data_small.csv' : 'kc_house_data.csv';

let {
  features: featuresA,
  labels: labelsA,
  testFeatures: testFeaturesA,
  testLabels: testLabelsA,
} = loadCSV(`./${csvFile}`, {
  shuffle: true,
  splitTest: testSize,
  dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
  labelColumns: ['price'],
});

const featuresT = tf.tensor(featuresA);
const labelsT = tf.tensor(labelsA);

for (let i = 0; i < testFeaturesA.length; i++) {
  const testFeatureRow = testFeaturesA[i];
  const testLabel = testLabelsA[i][0];
  runTestScenario(featuresT, labelsT, tf.tensor(testFeatureRow), testLabel);
}

/** HELPERS */
function runTestScenario(featuresT, labelsT, testFeaturesT, testLabel) {
  const result = tf_knn(featuresT, labelsT, testFeaturesT, k);
  const err = (testLabel - result) / testLabel;
  const percDiff = err * 100;

  let message = `⬇️ ${percDiff}% : test was low`;
  if (percDiff > 0) {
    message = `⬆️ ${percDiff}% : test was high`;
  }
  console.log(message);
}

require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

/** OPTIONS */
// ================================
const trainNumber = 60000;
const testNumber = 1000;
const iterations = 60;
const learningRate = 0.5;
const batchSize = 1000;
// ================================

function loadData(count) {
  const mnistData = mnist.training(0, count);
  const features = mnistData.images.values.map(image => _.flatMap(image));
  const labels = mnistData.labels.values.map(value => {
    const row = new Array(10).fill(0);
    row[parseInt(value)] = 1;
    return row;
  });
  return {
    features,
    labels,
  };
}
function createRegression() {
  const { features, labels } = loadData(trainNumber);
  const regression = new LogisticRegression(features, labels, {
    learningRate,
    iterations,
    batchSize,
  });
  return regression;
}
function plotHistory(history) {
  plot({
    x: regression.costHistory.reverse(),
  });
}

const regression = createRegression();

regression.train();
const mnistTestData = mnist.testing(0, testNumber);
const testFeatures = mnistTestData.images.values.map(image => _.flatMap(image));
const testEncodedLables = mnistTestData.labels.values.map(value => {
  const row = new Array(10).fill(0);
  row[parseInt(value)] = 1;
  return row;
});

const accuracy = regression.test(testFeatures, testEncodedLables);
console.log(`[M@][index] accuracy `, accuracy); // M@:

plotHistory(regression.costHistory);

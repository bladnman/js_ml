require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../utils/load-csv');
const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    // dataColumns: ['horsepower'],
    labelColumns: ['mpg'],
  }
);

const enableTesting = false;
const iterations = 30;

// set up an instance
const regression = new LinearRegression(features, labels);

doTrain(0.1, iterations);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error',
});

regression
  .predict([
    [120, 2, 380],
    [135, 2.1, 420],
  ])
  .print();

/** HELPERS */
// callable TRAINER
function doTrain(learningRate, iterations) {
  regression.options.learningRate = learningRate ?? 0.0001;
  regression.options.iterations = iterations ?? 100;
  regression.train();
  if (enableTesting) {
    const r2 = regression.test(testFeatures, testLabels);
    console.log(`[${learningRate}] R2 :`, r2);
  }
}

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

const iterations = 25;

// set up an instance
const regression = new LinearRegression(features, labels);
// train multiple values
// [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].forEach(learningRate =>
//   doTrain(learningRate)
// );
doTrain(0.1, iterations);

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error',
});

// callable TRAINER
function doTrain(learningRate, iterations) {
  regression.options.learningRate = learningRate ?? 0.0001;
  regression.options.iterations = iterations ?? 100;
  regression.train();
  const r2 = regression.test(testFeatures, testLabels);
  console.log(`[${learningRate}] R2 :`, r2);
}

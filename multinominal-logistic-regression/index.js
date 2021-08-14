require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('../utils/load-csv');
const LinearRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const LogisticRegression = require('./logistic-regression');
const _ = require('lodash');

const { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: 50,
    converters: {
      mpg: value => {
        const mpg = parseFloat(value);
        return [
          mpg < 15 ? 1 : 0,
          mpg >= 15 && mpg < 30 ? 1 : 0,
          mpg >= 30 ? 1 : 0,
        ];
      },
    },
  }
);

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5,
  iterations: 5,
  batchSize: 10,
  decisionBoundary: 0.5,
});
regression.train();
console.log(regression.test(testFeatures, _.flatMap(testLabels)));

plot({
  x: regression.costHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error',
});

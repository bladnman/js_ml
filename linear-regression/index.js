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
    // dataColumns: ['horsepower', 'weight', 'displacement'],
    dataColumns: ['horsepower'],
    labelColumns: ['mpg'],
  }
);
console.log(`[M@][index] starting`); // M@: logging
const regression = new LinearRegression(features, labels, {
  learningRate: 0.001,
  iterations: 10,
});

regression.train();
console.log(`[M@][index] m, b `, regression.m, regression.b); // M@: logging

// const r2 = regression.test(testFeatures, testLabels);

// plot({
//   x: regression.mseHistory.reverse(),
//   xLabel: 'Iteration #',
//   yLabel: 'Mean Squared Error',
// });

// console.log('R2 is', r2);

// regression.predict([[120, 2, 380]]).print();

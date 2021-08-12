require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

/**
 * Standardization with TensorFlow is simple, but uses
 * an internal method called `moments`
 */

// GIVEN SOME DATA
const features = tf.tensor([[10], [20], [35], [90]]);

// GET THE MEAN and VARIANCE
const { mean, variance } = tf.moments(features);

const standardized = features.sub(mean).div(variance.sqrt()); //?

console.log(`features`);
features.print();
/*
  [[10],
   [20],
   [35],
   [90]]
*/
console.log(`mean`);
mean.print();
/* 38.75 */

console.log(`variance`);
variance.print();
/* 954.6875 */
// this is the "squared std deviation"
// or "squared average distance to the mean"
// in this case STD DEV: 30.88...

console.log(`standardized`);
standardized.print();
/*
  [[-0.9304804],
   [-0.6068351],
   [-0.121367 ],
   [1.6586825 ]]
*/

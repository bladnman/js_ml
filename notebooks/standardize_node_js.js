require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const features = tf.ones([10, 1]);
features.print();

const { mean, variance } = tf.moments(features, 0);
mean.toString(); //?
variance.toString(); //?

const standardized = features.sub(mean).div(variance.pow(0.5));

standardized.print();

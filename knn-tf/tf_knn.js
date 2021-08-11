require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const tf_knn = (features, labels, predictionPoint, k) => {
  // standardization
  const { mean, variance } = tf.moments(features, 0);
  const standardizedPrediction = predictionPoint
    .sub(mean)
    .div(variance.pow(0.5));
  const standardizedFeatures = features.sub(mean).div(variance.pow(0.5));

  return (
    standardizedFeatures
      .sub(standardizedPrediction)
      .pow(2)
      .sum(1)
      .pow(0.5)
      // we have a [n] shape... need [n,1] like labels to concat
      .expandDims(1)
      // and we want to concat ACROSS not down
      .concat(labels, 1)
      .unstack()
      // here on out we are using JS array of tensor-rows
      .sort((a, b) => a.get(0) - b.get(0))
      .slice(0, k)
      .reduce((acc, tensor) => acc + tensor.get(1), 0) / k
  );
};
module.exports = tf_knn;

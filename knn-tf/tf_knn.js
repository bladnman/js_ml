const tf_knn = (features, labels, predictionPoint, k) => {
  return (
    features
      .sub(predictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      // we have a [4] shape... need [4,1] like labels to concap
      .expandDims(1)
      // and we want to concat ACROSS not down
      .concat(labels, 1)
      .unstack()
      // here on out we are using JS array
      .sort((a, b) => a.get(0) - b.get(0))
      .slice(0, k)
      .reduce((acc, tensor) => acc + tensor.get(1), 0) / k
  );
};
module.exports = tf_knn;

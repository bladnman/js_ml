const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression {
  constructor(features, labels, options) {
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, batchSize: 10 },
      options
    );

    this.labels = tf.tensor(labels);
    this.features = this.prepareFeatures(features);

    const numberOfWeights = this.features.shape[1];
    this.weights = tf.zeros([numberOfWeights, 1]);

    // mean squared error history
    this.mseHistory = [];
  }
  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights);
    const differences = currentGuesses.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const totalRows = this.features.shape[0];
    const batchSize = this.options.batchSize || totalRows;
    const batchQuantity = Math.floor(totalRows / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const featureBatch = this.features.slice(
          [batchSize * j, 0],
          [batchSize, -1]
        );
        const labelBatch = this.labels.slice(
          [batchSize * j, 0],
          [batchSize, -1]
        );
        this.gradientDescent(featureBatch, labelBatch);
      }

      this.recordMSE();
      this.updateLearningRate();
    }
  }

  /**
   * Will scale these features and apply an extra
   * column of 1's
   */
  prepareFeatures(features) {
    features = this.standardize(tf.tensor(features));
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    if (this.mean === undefined) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;
      this.variance = variance;
    }
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  predict(observations) {
    // this is how we get predictions
    return this.prepareFeatures(observations).matMul(this.weights);
  }

  test(testFeatures, testLabels) {
    // tensorize the labels
    testLabels = tf.tensor(testLabels);
    // get our predictions
    const predictions = predict(testFeatures);

    // Sum-of-Squares (SSres)
    const res = testLabels.sub(predictions).pow(2).sum().get();
    // Sum-of-Totals (SStot)
    const tot = testLabels.sub(testLabels.mean()).pow(2).sum().get();

    // Coeffecient of Determination
    const r2 = 1 - res / tot;

    /**
     * negative numbers here indicate our guesses are way off
     * we may as well just use the mean. adjust some things.
     */
    return r2;
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get();

    // put most recent on the front
    this.mseHistory.unshift(mse);
  }

  updateLearningRate() {
    if (this.mseHistory.length < 2) return;

    // MSE is increasing
    if (this.mseHistory[0] > this.mseHistory[1]) {
      this.options.learningRate /= 2;
    }

    // MSE trending down (good) -- let's go faster!
    else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LinearRegression;

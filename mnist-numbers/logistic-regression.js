const tf = require('@tensorflow/tfjs');

class LogisticRegression {
  constructor(featuresAA, labelsAA, options) {
    this.options = Object.assign(
      {
        learningRate: 0.1,
        iterations: 1000,
        batchSize: 10,
        decisionBoundary: 0.5,
      },
      options
    );

    this.labels = tf.tensor(labelsAA);
    this.features = this.prepareFeatures(featuresAA);

    const numberOfWeights = this.features.shape[1];
    this.weights = tf.zeros([numberOfWeights, this.labels.shape[1]]);

    // mean squared error history
    this.costHistory = [];
  }
  gradientDescent(features, labels) {
    let currentGuesses = features.matMul(this.weights);

    // to make this "logistic" we use sigmoid() or softmax()
    currentGuesses = currentGuesses.softmax();

    const differences = currentGuesses.sub(labels);
    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]);

    return this.weights.sub(slopes.mul(this.options.learningRate));
  }

  train() {
    const totalRows = this.features.shape[0];
    const batchSize = this.options.batchSize || totalRows;
    const batchQuantity = Math.floor(totalRows / batchSize);

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        this.weights = tf.tidy(() => {
          const featureBatch = this.features.slice(
            [batchSize * j, 0],
            [batchSize, -1]
          );
          const labelBatch = this.labels.slice(
            [batchSize * j, 0],
            [batchSize, -1]
          );
          return this.gradientDescent(featureBatch, labelBatch);
        });
      }

      this.recordCost();
      this.updateLearningRate();
    }
  }

  /**
   * Will scale these features and apply an extra
   * column of 1's
   */
  prepareFeatures(featuresAA) {
    let features = this.standardize(tf.tensor(featuresAA));
    features = tf.ones([features.shape[0], 1]).concat(features, 1);

    return features;
  }

  standardize(features) {
    if (this.mean === undefined) {
      const { mean, variance } = tf.moments(features, 0);
      this.mean = mean;

      // deal with 0-variances
      const filler = variance.cast('bool').logicalNot().cast('float32');

      this.variance = variance.add(filler);
    }

    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  predict(observationsAA) {
    // this is how we get predictions
    //   ->  matMul(this.weights)
    // to make this "logistic" we use softmax()
    return this.prepareFeatures(observationsAA)
      .matMul(this.weights)
      .softmax()
      .argMax(1);
  }

  test(testFeaturesAA, testLabelsAA) {
    const predictions = this.predict(testFeaturesAA);
    const testLabels = tf.tensor(testLabelsAA);

    // const incorrect = predictions.sub(testLabels).abs().sum().get();
    const incorrect = predictions.notEqual(testLabels.argMax(1)).sum().get();
    return (predictions.shape[0] - incorrect) / predictions.shape[0];
  }

  recordCost() {
    // clean up our memory (tensors)
    const cost = tf.tidy(() => {
      const guesses = this.features.matMul(this.weights).softmax();
      const termOne = this.labels.transpose().matMul(guesses.add(1e-7).log());
      const termTwo = this.labels
        .mul(-1)
        .add(1)
        .transpose()
        .matMul(guesses.mul(-1).add(1).add(1e-7).log());
      // adding a tiny constant (1e-7) to avoid log(0)

      return termOne.add(termTwo).div(this.features.shape[0]).mul(-1).get(0, 0);
    });
    // put most recent on the front
    this.costHistory.unshift(cost);
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) return;

    // MSE is increasing
    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2;
    }

    // MSE trending down (good) -- let's go faster!
    else {
      this.options.learningRate *= 1.05;
    }
  }
}

module.exports = LogisticRegression;

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
    this.weights = tf.zeros([numberOfWeights, 1]);

    // mean squared error history
    this.costHistory = [];
  }
  gradientDescent(features, labels) {
    let currentGuesses = features.matMul(this.weights);

    // to make this "logistic" we use sigmoid()
    currentGuesses = currentGuesses.sigmoid();

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
      this.variance = variance;
    }
    return features.sub(this.mean).div(this.variance.pow(0.5));
  }

  predict(observationsAA) {
    // this is how we get predictions
    //   ->  matMul(this.weights)
    // to make this "logistic" we use sigmoid()
    return this.prepareFeatures(observationsAA)
      .matMul(this.weights)
      .sigmoid()
      .greater(this.options.decisionBoundary)
      .cast('float32');
  }

  test(testFeaturesAA, testLabelsAA) {
    const predicitons = this.predict(testFeaturesAA);
    const testLabels = tf.tensor(testLabelsAA);
    const incorrect = predicitons.sub(testLabels).abs().sum().get();
    return (predicitons.shape[0] - incorrect) / predicitons.shape[0];
  }

  recordCost() {
    const guesses = this.features.matMul(this.weights).sigmoid();
    const termOne = this.labels.transpose().matMul(guesses.log());
    const termTwo = this.labels
      .mul(-1)
      .add(1)
      .transpose()
      .matMul(guesses.mul(-1).add(1).log());

    const cost = termOne
      .add(termTwo)
      .div(this.features.shape[0])
      .mul(-1)
      .get(0, 0);

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

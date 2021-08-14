require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

function test(features, labels, predictions) {
  labels = tf.tensor(labels);
  predictions = tf.tensor(predictions);
  const diffs = predictions.sub(labels).abs();

  diffs.print();
  const incorrect = diffs.sum().get();
  return incorrect / labels.shape[0];
}

const quality = test(null, [1, 0, 1, 0], [1, 1, 0, 0]);
quality;

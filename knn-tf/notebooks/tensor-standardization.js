require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

function do1() {
  const numbers = tf.tensor([
    [1, 2],
    [3, 4],
    [5, 6],
  ]);
  const { mean, variance } = tf.moments(numbers, 0);
  const deviation = numbers.sub(mean).div(variance.pow(0.5));
  deviation.print();
}

/**
 * --------------------------------
 *    MAINS
 * --------------------------------
 */
do1();

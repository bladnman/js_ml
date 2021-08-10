require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');
const tf_knn = require('./tf_knn');

const bob = 3; //?

let { features, labels, testFeatures, testLabels } = loadCSV(
  'kc_house_data.csv',
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'sqft_lot', 'sqft_living'],
    labelColumns: ['price'],
  }
);

features = tf.tensor(features);
labels = tf.tensor(labels); //?

testFeatures.forEach((testPoint, i) => {
  const result = tf_knn(features, labels, tf.tensor(testPoint), 10);
  const err = (testLabels[i][0] - result) / testLabels[i][0];
  console.log('Error', err * 100);
});

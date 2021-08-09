require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

function do1D() {
  const data1 = tf.tensor([1, 2, 3]);
  const data2 = tf.tensor([4, 5, 6]);
  console.log('add', data1.add(data1).toString());
  console.log('sub', data1.sub(data1).toString());
  console.log('mul', data1.mul(data1).toString());
  console.log('div', data1.div(data1).toString());
  console.log('less', data1.less(data1).toString());
  console.log('greater', data1.greater(data1).toString());
}

function do2D() {
  // multi-dimentional as well
  const data1 = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const data2 = tf.tensor([
    [7, 8, 9],
    [10, 11, 12],
  ]);
  console.log('add', data1.add(data1).toString());
  console.log('sub', data1.sub(data1).toString());
  console.log('mul', data1.mul(data1).toString());
  console.log('div', data1.div(data1).toString());
  console.log('less', data1.less(data1).toString());
  console.log('greater', data1.greater(data1).toString());
}

function doMisshapen1D() {
  const data1 = tf.tensor([1, 2, 3, 4]);
  const data2 = tf.tensor([10]);
  console.log('add', data1.add(data2).toString());
  console.log('sub', data1.sub(data2).toString());
  console.log('mul', data1.mul(data2).toString());
  console.log('div', data1.div(data2).toString());
  console.log('less', data1.less(data2).toString());
  console.log('greater', data1.greater(data2).toString());
}

function doMisshapen2D() {
  const data1 = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const data2 = tf.tensor([10]);
  console.log('add', data1.add(data2).toString());
  console.log('sub', data1.sub(data2).toString());
  console.log('mul', data1.mul(data2).toString());
  console.log('div', data1.div(data2).toString());
  console.log('less', data1.less(data2).toString());
  console.log('greater', data1.greater(data2).toString());
}

function gettingAtTheData() {
  const data2d = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  console.log(data2d.get(1, 2));

  const dataXd = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18],
  ]);
  const rowIndex = 0;
  const columnIndex = 1;
  // ex : dataXd.slice([0, 1], [-1, 1])
  console.log(dataXd.slice([rowIndex, columnIndex], [-1, 1]).toString());
}
function doConcat() {
  const data1 = tf.tensor([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  const data2 = tf.tensor([
    [7, 8, 9],
    [10, 11, 12],
  ]);

  // concat down (vertically)
  const concatDown = data1.concat(data2);
  concatDown.shape; //?
  console.log(concatDown.toString());

  // concat across (horizontally)
  const concatAcross = data1.concat(data2, 1);
  concatAcross.shape; //?
  console.log(concatAcross.toString());
}
/**
 * --------------------------------
 *    MAINS
 * --------------------------------
 */
do1D();
do2D();
doMisshapen1D();
doMisshapen2D();
gettingAtTheData();
doConcat();

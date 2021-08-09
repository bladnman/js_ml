const tf = require('@tensorflow/tfjs');

const jumpData = tf.tensor([
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
]);
const playerData = tf.tensor([
  [1, 160],
  [2, 160],
  [3, 160],
  [4, 160],
]);
function doItKeepingDims() {
  /**
   * sum will remove our dims and then we cannot concat
   * jumpSum to playerData as we want. The `sum` function
   * however, takes an argument that prevents these dims from
   * being lost.
   */
  const jumpSum = jumpData.sum(1, true); // sum horiz
  console.log(jumpSum.toString());

  const playerAndTotalJumpData = jumpSum.concat(playerData, 1);
  console.log(playerAndTotalJumpData.toString());
}
function doIt() {
  /**
   * there is an `expandDims` function that will add
   * another "column" for us (using 1 param)
   */
  const jumpSum = jumpData.sum(1).expandDims(1); // sum horiz
  console.log(jumpSum.toString());

  const playerAndTotalJumpData = jumpSum.concat(playerData, 1);
  console.log(playerAndTotalJumpData.toString());
}

/**
 * --------------------------------
 *    MAINS
 * --------------------------------
 */
doItKeepingDims();
doIt();

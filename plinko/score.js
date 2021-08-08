const outputs = [];

const predictionPoint = 300;
const K = 5;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  for (let i = 2; i < 6; i++) {
    const mostCommonBucket = mostCommonBucket_knn(outputs, i);
    console.log(`[M@][${i}] `, mostCommonBucket); // M@: logging
  }
}

function mostCommonBucket_knn(outputs, k = 3) {
  const distanceOutputs = simplifyByDistance(outputs, k); //?
  const countsPerBucket = _.countBy(distanceOutputs, (row) => row[1]); //?
  const mostCommonBucket = _.chain(countsPerBucket)
    .toPairs()
    .sortBy((row) => row[1])
    .last()
    .first()
    .parseInt()
    .value(); //?
  return mostCommonBucket;
}

function simplifyByDistance(outputs, k = 3) {
  return _.chain(outputs)
    .map((row) => [distance(row[0]), row[3]])
    .sortBy((row) => row[0])
    .slice(0, k)
    .value();
}
function distance(point) {
  return Math.abs(point - predictionPoint);
}

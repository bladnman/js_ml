const _ = require('lodash');

const outputs = [
  [258, 0.5126933228128626, 16, 1],
  [367, 0.5204384318451813, 16, 4],
  [216, 0.5193340093282988, 16, 4],
  [230, 0.5127335530980701, 16, 4],
  [389, 0.5458797279524231, 16, 3],
];
const predictionPoint = 300;
const k = 3;

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

const distanceOutputs = simplifyByDistance(outputs, k); //?
const countsPerBucket = _.countBy(distanceOutputs, (row) => row[1]); //?
const mostCommonBucket = _.chain(countsPerBucket)
  .toPairs()
  .sortBy((row) => row[1])
  .last()
  .first()
  .parseInt()
  .value(); //?

const _ = require('lodash');
const numbers = [
  [10, 5],
  [17, 2],
  [34, 1],
  [60, -5],
];
const sorted = _.sortBy(numbers, (row) => row[1]); //?
const mapped = sorted.map((row) => row[1]); //?

const chained = _.chain(numbers)
  .sortBy((row) => row[1])
  .map((row) => row[1])
  .value(); //?

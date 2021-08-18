const fs = require('fs');
const _ = require('lodash');
const shuffleSeed = require('shuffle-seed');

function loadCSV(filename, options) {
  let data = fs.readFileSync(filename, { encoding: 'utf-8' });
  data = data.split('\n').map(row => row.split(','));
  data = data.map(row => _.dropRightWhile(row, val => val === ''));
  const headers = _.first(data);

  data = data.map((row, index) => {
    // skip headers
    if (index === 0) return row;
    // get values
    return row.map((element, index) => {
      const result = parseFloat(element);
      return _.isNaN(result) ? element : result;
    });
  });

  headers;
  data;
}

loadCSV('data.csv');

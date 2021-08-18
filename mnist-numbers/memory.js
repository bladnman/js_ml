// run with
//     node --inspect-brk memory.js

const _ = require('lodash');

function loadData() {
  const randoms = _.range(0, 999999);
  return randoms;
}

const data = loadData();

debugger;

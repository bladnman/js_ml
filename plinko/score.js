const outputs = [];

const predictionPoint = 300;
const K = 5;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 100;
  const k = 10;

  _.range(0, 3).forEach(feature => {
    const data = _.map(outputs, row => [row[feature], _.last(row)]);
    const [testSet, trainingSet] = splitDataset(
      minMaxFeatures(data, 1),
      testSetSize
    );

    const accuracy = _.chain(testSet)
      // only get rows where the bucket matched the assumption
      .filter(
        testObservation =>
          knn(trainingSet, _.initial(testObservation), k) ===
          _.last(testObservation)
      )
      .size()
      .divide(testSetSize)
      .value();

    console.log(`fearture ${feature} => Accuracy:`, accuracy);
  });
}

function knn(data, point, k) {
  const nnData = _.chain(data)
    // create data items that have [distance_from_start, bucket]
    // .map(row => [distance(row[0], point), row[3]])
    .map(row => {
      return [multiDimDistance(_.initial(row), point), _.last(row)];
    })
    // order by distance_from_start
    .sortBy(row => row[0])
    // take the k nearest neighbors
    .slice(0, k)
    .value();

  return (
    _.chain(nnData)
      // all of this sums which bucket was hit most
      // countBy creates has like {"bucket", count, "bucket2", count2}
      .countBy(row => row[1])
      // will create array of arrays of this hash
      // [["bucket1", count1], ["bucket2", count2]]
      .toPairs()
      // orders the counts ASC on the count property
      .sortBy(row => row[1])
      // takes final (highest) item which is still an array
      // ["bucket", count], so take last "row"
      .last()
      // get first bucket string
      .first()
      // make it a number
      .parseInt()
      .value()
  );
}
function distance(pointA, pointB) {
  return Math.abs(pointA - pointB);
}
function multiDimDistance(pointA, pointB) {
  // ex : dataA = [300, .5, 16]
  return (
    _.chain(pointA)
      .zip(pointB)
      .map(([a, b]) => (a - b) ** 2)
      .sum()
      .value() ** 0.5
  );
}
function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data);

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount);

  return [testSet, trainingSet];
}

function minMaxFeatures(data, featureCount) {
  const clonedData = _.cloneDeep(data);

  for (let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i]);
    const min = _.min(column);
    const max = _.max(column);
    for (let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min);
    }
  }

  return clonedData;
}

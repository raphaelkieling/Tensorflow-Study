const model = tf.sequential();

// Create a Layer
// Dense is a full connected layer
const hidden = tf.layers.dense({
  units: 2,
  inputShape: [2],
  activation: "sigmoid"
});

const output = tf.layers.dense({
  units: 1,
  activation: "sigmoid"
});

// Add the layers
model.add(hidden);
model.add(output);

// Configure the model
model.compile({
  optimizer: tf.train.sgd(0.5),
  loss: "meanSquaredError"
});

let train_data = [
  {
    input: [0, 0],
    output: [0]
  },
  {
    input: [0, 1],
    output: [1]
  },
  {
    input: [1, 0],
    output: [1]
  },
  {
    input: [1, 1],
    output: [0]
  }
];

async function getSanitizedData() {
  let train_x = train_data.map(data => data.input);
  let train_y = train_data.map(data => data.output);

  return { train_x: tf.tensor(train_x), train_y: tf.tensor(train_y) };
}

async function train() {
  let { train_x, train_y } = await getSanitizedData();

  for (let index = 0; index < 1000; index++) {
    let response = await model.fit(train_x, train_y, {
      shuffle: true,
      epochs: 10
    });
    console.log("Loss: ", response.history.loss[0]);
  }
}

train().then(async _ => {
  console.log("Finished Training");

  let { train_x } = await getSanitizedData();

  model
    .predict(train_x)
    .round()
    .print();
});

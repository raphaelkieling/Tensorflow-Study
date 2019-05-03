const model = tf.sequential();

// Create a Layer
// Dense is a full connected layer
const hidden = tf.layers.dense({
  units: 4,
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

/**
 * [0,0]     => 1
 * [0.5,0.5] => 0.5
 * [1,1]     => 0
 */
const train_x = tf.tensor2d([[0, 0], [0.5, 0.5], [1, 1]]);
const train_y = tf.tensor2d([[1], [0.5], [0]]);

async function train() {
  for (let index = 0; index < 100; index++) {
    let response = await model.fit(train_x, train_y, {
      shuffle: true,
      epochs: 10
    });
    console.log(response.history.loss[0]);
  }
}

train().then(_ => {
  console.log("Training complete");
  model.predict(train_x).print();
});

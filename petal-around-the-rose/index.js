class DicesAroundTheRose {
  constructor() {
    this.dices = [[1, 0], [2, 0], [3, 2], [4, 0], [5, 4], [6, 0]];
  }

  getFiveDices() {
    let dicesToShow = [];

    let getRandomDice = function(dices) {
      return dices[Math.round(Math.random() * (dices.length - 1))];
    };

    for (let index = 0; index < 5; index++) {
      dicesToShow.push(getRandomDice(this.dices));
    }

    return dicesToShow;
  }

  isCorrect(dices, petal) {
    return !!dices.find(
      ([number, petals]) => number === dice && petals === petal
    );
  }
}

// The nework receive to TRAIN [numberDices, is correct]
// The nework receive to PREDICT [numberDices]
class NeuronNetworkAroundTheRose {
  constructor() {
    this.model = this.createModel();
    this.learningRate = 0.01;
  }

  createModel() {
    return tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [5], units: 4 }),
        tf.layers.dense({ units: 1 })
      ]
    });
  }

  compile() {
    return this.model.compile({
      optimizer: tf.train.sgd(this.learningRate),
      loss: "meanSquaredError"
    });
  }

  async save() {
    return await this.model.save("localstorage://petal-aroud-the-rouse");
  }

  async load() {
    this.model = await tf
      .loadLayersModel("localstorage://petal-aroud-the-rouse")
      .catch(err => console.log("Not have model"));

    return this;
  }
}

let brain = new NeuronNetworkAroundTheRose();
let dices = new DicesAroundTheRose();

async function start() {
  let dices = new DicesAroundTheRose();
  let brain = new NeuronNetworkAroundTheRose();

  await brain.load();

  await brain.compile();

  for (let index = 0; index < 100; index++) {
    let dicesToPlay = dices.getFiveDices();
    let train_x = tf.tensor([dicesToPlay.map(n => n[0])]);
    let train_y = tf.tensor([
      [dicesToPlay.reduce((ant, curr) => ant + curr[1], 0)]
    ]);

    train_x.print();
    train_y.print();
    console.log("----------");

    await brain.model
      .fit(train_x, train_y, {
        shuffle: true,
        epoch: 5
      })
      .then(h => {
        console.log("[Loss]: " + h.history.loss[0]);
      });
  }

  await brain.save();

  //   Predict
  let diceToPredict = dices.getFiveDices();
  let toPredict = tf.tensor([diceToPredict.map(n => n[0])]);

  console.log(diceToPredict);
  brain.model.predict(toPredict).print();
}

start();

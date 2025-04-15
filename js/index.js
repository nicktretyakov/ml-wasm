import init, { NeuralNetwork, init_panic_hook } from "../pkg/ml_wasm.js"

async function run() {
  await init()
  init_panic_hook()

  const nn = new NeuralNetwork(2, 4, 1, 0.5)

  // XOR dataset
  const training_data = [
    { inputs: [0, 0], targets: [0] },
    { inputs: [0, 1], targets: [1] },
    { inputs: [1, 0], targets: [1] },
    { inputs: [1, 1], targets: [0] },
  ]

  // Training
  console.log("Starting training...")
  for (let epoch = 0; epoch < 10000; epoch++) {
    for (const data of training_data) {
      nn.train(data.inputs, data.targets)
    }

    // Log progress every 1000 epochs
    if (epoch % 1000 === 0) {
      console.log(`Epoch ${epoch}`)
    }
  }
  console.log("Training complete!")

  // Testing
  console.log("[0,0] =>", nn.predict([0, 0])) // ≈0
  console.log("[0,1] =>", nn.predict([0, 1])) // ≈1
  console.log("[1,0] =>", nn.predict([1, 0])) // ≈1
  console.log("[1,1] =>", nn.predict([1, 1])) // ≈0
}

run().catch(console.error)

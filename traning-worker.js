import init, { NeuralNetwork, init_panic_hook } from "./pkg/ml_wasm.js";

let neuralNetwork = null;

// Initialize the WebAssembly module
async function initWasm() {
  try {
    await init();
    init_panic_hook();
    return true;
  } catch (error) {
    postMessage({
      type: "error",
      data: { error: "Failed to initialize WebAssembly module" },
    });
    return false;
  }
}

// Handle messages from the main thread
self.onmessage = async (e) => {
  const { type, data } = e.data;

  if (type === "train") {
    // Initialize WASM if not already done
    if (!(await initWasm())) return;

    const { trainingData, epochs, hiddenNeurons, learningRate } = data;

    try {
      // Create a new neural network
      neuralNetwork = new NeuralNetwork(2, hiddenNeurons, 1, learningRate);

      // Train the network
      for (let epoch = 0; epoch < epochs; epoch++) {
        for (const data of trainingData) {
          neuralNetwork.train(data.inputs, data.targets);
        }

        // Report progress every 100 epochs or at the end
        if (epoch % 100 === 0 || epoch === epochs - 1) {
          postMessage({
            type: "progress",
            data: { epoch, epochs },
          });
        }
      }

      // Training complete
      postMessage({ type: "complete" });
    } catch (error) {
      postMessage({
        type: "error",
        data: { error: error.toString() },
      });
    }
  }
};

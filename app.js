import init, { NeuralNetwork, init_panic_hook } from "./pkg/ml_wasm.js";

// Global variables
let neuralNetwork = null;
let isTraining = false;
let trainingWorker = null;
let vizContext = null;

// DOM elements
const loadingOverlay = document.getElementById("loading-overlay");
const createNetworkBtn = document.getElementById("create-network");
const trainNetworkBtn = document.getElementById("train-network");
const testNetworkBtn = document.getElementById("test-network");
const hiddenNeuronsInput = document.getElementById("hidden-neurons");
const learningRateInput = document.getElementById("learning-rate");
const learningRateValue = document.getElementById("learning-rate-value");
const epochsInput = document.getElementById("epochs");
const trainingProgress = document.getElementById("training-progress");
const progressText = document.getElementById("progress-text");
const testInput1 = document.getElementById("test-input-1");
const testInput2 = document.getElementById("test-input-2");
const outputValue = document.getElementById("output-value");
const xorViz = document.getElementById("xor-viz");

// XOR dataset
const trainingData = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

// Initialize the application
async function initApp() {
  try {
    // Load the WebAssembly module
    await init();
    init_panic_hook();

    // Setup event listeners
    setupEventListeners();

    // Initialize visualization
    initVisualization();

    // Hide loading overlay
    loadingOverlay.style.display = "none";
  } catch (error) {
    console.error("Failed to initialize WebAssembly module:", error);
    alert(
      "Failed to load the neural network. Please check the console for details.",
    );
  }
}

// Setup event listeners
function setupEventListeners() {
  // Learning rate slider
  learningRateInput.addEventListener("input", () => {
    learningRateValue.textContent = learningRateInput.value;
  });

  // Create network button
  createNetworkBtn.addEventListener("click", createNetwork);

  // Train network button
  trainNetworkBtn.addEventListener("click", startTraining);

  // Test network button
  testNetworkBtn.addEventListener("click", testNetwork);
}

// Create a new neural network
function createNetwork() {
  const hiddenNeurons = Number.parseInt(hiddenNeuronsInput.value);
  const learningRate = Number.parseFloat(learningRateInput.value);

  try {
    neuralNetwork = new NeuralNetwork(2, hiddenNeurons, 1, learningRate);
    trainNetworkBtn.disabled = false;

    // Update UI
    createNetworkBtn.textContent = "Network Created";
    setTimeout(() => {
      createNetworkBtn.textContent = "Recreate Network";
    }, 2000);

    // Reset visualization
    drawDecisionBoundary();
  } catch (error) {
    console.error("Failed to create neural network:", error);
    alert(
      "Failed to create the neural network. Please check the console for details.",
    );
  }
}

// Start training the network
function startTraining() {
  if (!neuralNetwork || isTraining) return;

  const epochs = Number.parseInt(epochsInput.value);
  isTraining = true;

  // Update UI
  trainNetworkBtn.disabled = true;
  createNetworkBtn.disabled = true;
  testNetworkBtn.disabled = true;
  trainingProgress.style.width = "0%";
  progressText.textContent = "0%";

  // Create a worker for training to avoid blocking the UI
  if (window.Worker) {
    // If we already have a worker, terminate it
    if (trainingWorker) {
      trainingWorker.terminate();
    }

    // Create a new worker
    trainingWorker = new Worker("./training-worker.js", { type: "module" });

    // Listen for messages from the worker
    trainingWorker.onmessage = (e) => {
      const { type, data } = e.data;

      if (type === "progress") {
        // Update progress bar
        const progress = Math.round((data.epoch / epochs) * 100);
        trainingProgress.style.width = `${progress}%`;
        progressText.textContent = `${progress}%`;

        // Update visualization every 10% progress
        if (progress % 10 === 0) {
          // Get predictions for visualization
          updateVisualization();
        }
      } else if (type === "complete") {
        // Training complete
        finishTraining();
      } else if (type === "error") {
        console.error("Training error:", data.error);
        alert(
          "An error occurred during training. Please check the console for details.",
        );
        finishTraining();
      }
    };

    // Start the worker with training data
    trainingWorker.postMessage({
      type: "train",
      data: {
        trainingData,
        epochs,
        hiddenNeurons: Number.parseInt(hiddenNeuronsInput.value),
        learningRate: Number.parseFloat(learningRateInput.value),
      },
    });
  } else {
    // Fallback for browsers that don't support Web Workers
    alert(
      "Your browser does not support Web Workers. Training might freeze the UI.",
    );
    trainWithoutWorker(epochs);
  }
}

// Train without using a Web Worker (fallback)
function trainWithoutWorker(epochs) {
  let currentEpoch = 0;

  function trainBatch() {
    const batchSize = 100; // Train 100 epochs at a time to avoid blocking UI
    const startEpoch = currentEpoch;
    const endEpoch = Math.min(currentEpoch + batchSize, epochs);

    for (let epoch = startEpoch; epoch < endEpoch; epoch++) {
      for (const data of trainingData) {
        neuralNetwork.train(data.inputs, data.targets);
      }
      currentEpoch++;
    }

    // Update progress
    const progress = Math.round((currentEpoch / epochs) * 100);
    trainingProgress.style.width = `${progress}%`;
    progressText.textContent = `${progress}%`;

    // Update visualization every 10% progress
    if (progress % 10 === 0) {
      updateVisualization();
    }

    // Continue training or finish
    if (currentEpoch < epochs) {
      setTimeout(trainBatch, 0);
    } else {
      finishTraining();
    }
  }

  // Start training
  trainBatch();
}

// Finish training and update UI
function finishTraining() {
  isTraining = false;

  // Update UI
  trainNetworkBtn.disabled = false;
  createNetworkBtn.disabled = false;
  testNetworkBtn.disabled = false;
  trainingProgress.style.width = "100%";
  progressText.textContent = "100%";

  // Update visualization
  updateVisualization();

  // Enable testing
  testNetworkBtn.disabled = false;
}

// Test the network with custom inputs
function testNetwork() {
  if (!neuralNetwork) return;

  const input1 = Number.parseFloat(testInput1.value);
  const input2 = Number.parseFloat(testInput2.value);

  try {
    const output = neuralNetwork.predict([input1, input2]);
    outputValue.textContent = output[0].toFixed(4);

    // Highlight the result based on the output value
    if (output[0] >= 0.5) {
      outputValue.style.color = "blue";
    } else {
      outputValue.style.color = "red";
    }
  } catch (error) {
    console.error("Prediction error:", error);
    outputValue.textContent = "Error";
    outputValue.style.color = "var(--error-color)";
  }
}

// Initialize the visualization canvas
function initVisualization() {
  vizContext = xorViz.getContext("2d");

  // Draw the initial state
  drawDecisionBoundary();
}

// Update the visualization with current network state
function updateVisualization() {
  if (!neuralNetwork || !vizContext) return;

  drawDecisionBoundary();
}

// Draw the decision boundary
function drawDecisionBoundary() {
  const width = xorViz.width;
  const height = xorViz.height;
  const resolution = 50; // Number of points to sample in each dimension
  const ctx = vizContext;

  // Clear the canvas
  ctx.clearRect(0, 0, width, height);

  // Draw the background grid
  ctx.strokeStyle = "#ddd";
  ctx.lineWidth = 1;

  // Draw grid lines
  for (let i = 0; i <= 10; i++) {
    const pos = (i / 10) * width;

    // Vertical lines
    ctx.beginPath();
    ctx.moveTo(pos, 0);
    ctx.lineTo(pos, height);
    ctx.stroke();

    // Horizontal lines
    ctx.beginPath();
    ctx.moveTo(0, pos);
    ctx.lineTo(width, pos);
    ctx.stroke();
  }

  // Draw the decision boundary if we have a neural network
  if (neuralNetwork) {
    // Create an image data to visualize the decision boundary
    const imageData = ctx.createImageData(width, height);
    const data = imageData.data;

    // For each pixel in the canvas
    for (let x = 0; x < width; x++) {
      for (let y = 0; y < height; y++) {
        // Convert canvas coordinates to input space (0 to 1)
        const input1 = x / width;
        const input2 = y / height;

        // Get the prediction
        const output = neuralNetwork.predict([input1, input2])[0];

        // Calculate the index in the image data array
        const idx = (y * width + x) * 4;

        // Set the color based on the output (red for 0, blue for 1)
        if (output < 0.5) {
          // Red (output close to 0)
          data[idx] = 255;
          data[idx + 1] = 0;
          data[idx + 2] = 0;
          data[idx + 3] = Math.max(30, 200 * (1 - output * 2)); // Alpha based on confidence
        } else {
          // Blue (output close to 1)
          data[idx] = 0;
          data[idx + 1] = 0;
          data[idx + 2] = 255;
          data[idx + 3] = Math.max(30, 200 * ((output - 0.5) * 2)); // Alpha based on confidence
        }
      }
    }

    // Put the image data on the canvas
    ctx.putImageData(imageData, 0, 0);
  }

  // Draw the training data points
  ctx.lineWidth = 2;

  for (const data of trainingData) {
    const x = data.inputs[0] * width;
    const y = data.inputs[1] * height;
    const output = data.targets[0];

    // Draw a circle for each training point
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = output === 0 ? "red" : "blue";
    ctx.fill();
    ctx.strokeStyle = "white";
    ctx.stroke();
  }

  // Draw the axes labels
  ctx.fillStyle = "black";
  ctx.font = "12px Arial";
  ctx.textAlign = "center";

  // X-axis labels
  ctx.fillText("0", 10, height - 5);
  ctx.fillText("1", width - 10, height - 5);
  ctx.fillText("Input 1", width / 2, height - 5);

  // Y-axis labels
  ctx.textAlign = "right";
  ctx.fillText("0", 15, height - 10);
  ctx.fillText("1", 15, 15);
  ctx.save();
  ctx.translate(15, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Input 2", 0, 0);
  ctx.restore();
}

// Start the application
initApp();

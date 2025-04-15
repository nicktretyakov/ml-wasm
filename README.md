# Neural Network in Rust + WebAssembly

This project implements a simple neural network in Rust that can be compiled to WebAssembly (WASM) and used in JavaScript. The neural network is trained to solve the XOR problem.

## Features

- Simple feedforward neural network with one hidden layer
- Matrix operations using ndarray
- WebAssembly compilation for use in web browsers
- JavaScript interface

## Prerequisites

- [Rust](https://www.rust-lang.org/tools/install)
- [wasm-pack](https://rustwasm.github.io/wasm-pack/installer/)
- A web server for serving the HTML/JS files

## Building

 Build the WebAssembly module:
   \`\`\`
   wasm-pack build --target web
   \`\`\`

 Serve the project directory with a web server:
   \`\`\`
   # Using Python's built-in server
   python -m http.server
   # Or using Node.js with http-server
   npx http-server
   \`\`\`

 Open your browser and navigate to `http://localhost:8000` (or whatever port your server is using).

## How it Works

The neural network is implemented in Rust with the following components:

- `NeuralNetwork` struct: Represents the neural network with weights and biases
- `predict` method: Forward pass through the network
- `train` method: Backpropagation to update weights and biases

The JavaScript code loads the WASM module, creates a neural network, trains it on the XOR dataset, and tests it.

## Customizing

You can modify the neural network architecture by changing the parameters when creating a new `NeuralNetwork`:

\`\`\`javascript
// new NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
const nn = new NeuralNetwork(2, 4, 1, 0.5);
\`\`\`

## License

MIT
\`\`\`

Let's create a .gitignore file:

```gitignore file=".gitignore"
/target
/pkg
/node_modules
Cargo.lock

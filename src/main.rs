use ml_wasm::NeuralNetwork;

fn main() {
    println!("Neural Network XOR Example");

    // Create a new neural network with 2 inputs, 4 hidden neurons, 1 output, and learning rate 0.5
    let mut nn = NeuralNetwork::new(2, 4, 1, 0.5);

    // XOR dataset
    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    // Training
    println!("Starting training...");
    for epoch in 0..10000 {
        for (inputs, targets) in &training_data {
            nn.train(inputs, targets);
        }

        // Log progress every 1000 epochs
        if epoch % 1000 == 0 {
            println!("Epoch {}", epoch);
        }
    }
    println!("Training complete!");

    // Testing
    println!("\nTesting the trained network:");
    for (inputs, expected) in &training_data {
        let prediction = nn.predict(inputs);
        println!(
            "Input: [{}, {}], Prediction: {:.4}, Expected: {}",
            inputs[0], inputs[1], prediction[0], expected[0]
        );
    }
}

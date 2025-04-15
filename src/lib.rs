#![allow(non_snake_case)]

use ndarray::{Array1, Array2};
use rand::Rng;
use std::f64::consts::E;

// Only include wasm-bindgen when targeting wasm32
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

// Define the struct with conditional wasm_bindgen attribute
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
pub struct NeuralNetwork {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_ih: Array2<f64>,
    weights_ho: Array2<f64>,
    bias_h: Array1<f64>,
    bias_o: Array1<f64>,
    learning_rate: f64,
}

// Implementation with conditional wasm_bindgen attribute
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl NeuralNetwork {
    // Constructor with conditional wasm_bindgen attribute
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen(constructor))]
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();
        let weights_ih =
            Array2::from_shape_fn((hidden_size, input_size), |_| rng.gen_range(-1.0..1.0));
        let weights_ho =
            Array2::from_shape_fn((output_size, hidden_size), |_| rng.gen_range(-1.0..1.0));
        let bias_h = Array1::from_shape_fn(hidden_size, |_| rng.gen_range(-1.0..1.0));
        let bias_o = Array1::from_shape_fn(output_size, |_| rng.gen_range(-1.0..1.0));

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            learning_rate,
        }
    }

    fn sigmoid(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|v| 1.0 / (1.0 + E.powf(-v)))
    }

    fn sigmoid_derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x * (1.0 - x)
    }

    // Predict method with conditional wasm_bindgen attribute
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        let inputs = Array2::from_shape_vec((1, self.input_size), inputs.to_vec())
            .expect("Invalid input shape");
        let hidden = self.sigmoid(&(inputs.dot(&self.weights_ih.t()) + &self.bias_h));
        let outputs = self.sigmoid(&(hidden.dot(&self.weights_ho.t()) + &self.bias_o));
        outputs.into_raw_vec()
    }

    // Train method with conditional wasm_bindgen attribute
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
    pub fn train(&mut self, inputs: &[f64], targets: &[f64]) {
        let inputs = Array2::from_shape_vec((1, self.input_size), inputs.to_vec())
            .expect("Invalid input shape");
        let targets = Array2::from_shape_vec((1, self.output_size), targets.to_vec())
            .expect("Invalid target shape");

        // Forward pass
        let hidden_inputs = inputs.dot(&self.weights_ih.t()) + &self.bias_h;
        let hidden_outputs = self.sigmoid(&hidden_inputs);
        let final_inputs = hidden_outputs.dot(&self.weights_ho.t()) + &self.bias_o;
        let final_outputs = self.sigmoid(&final_inputs);

        // Backpropagation
        let output_errors = &targets - &final_outputs;
        let output_gradients = output_errors * self.sigmoid_derivative(&final_outputs);
        let hidden_errors = output_gradients.dot(&self.weights_ho);
        let hidden_gradients = hidden_errors * self.sigmoid_derivative(&hidden_outputs);

        // Update weights and biases
        self.weights_ho =
            &self.weights_ho + &(output_gradients.t().dot(&hidden_outputs) * self.learning_rate);
        self.weights_ih =
            &self.weights_ih + &(hidden_gradients.t().dot(&inputs) * self.learning_rate);

        // Fix the bias updates by using element-wise operations
        let output_delta = &output_gradients.sum_axis(ndarray::Axis(0)) * self.learning_rate;
        let hidden_delta = &hidden_gradients.sum_axis(ndarray::Axis(0)) * self.learning_rate;

        for i in 0..self.bias_o.len() {
            self.bias_o[i] += output_delta[i];
        }

        for i in 0..self.bias_h.len() {
            self.bias_h[i] += hidden_delta[i];
        }
    }
}

// Only include init_panic_hook when targeting wasm32
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

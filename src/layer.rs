use snippets;

const WEIGHT_BOUNDRY: f32 = 2.0;

#[derive(Clone)]
pub struct Layer {
    pub num_nodes_in: usize,
    pub num_nodes_out: usize,
    pub cost_gradient_weights: Vec<Vec<f32>>,
    pub cost_gradient_biases: Vec<f32>,
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
}

impl Layer {
    // Create the layer.
    pub fn new(num_nodes_in: usize, num_nodes_out: usize) -> Self {
        let cost_gradient_weights = vec![vec![0.0; num_nodes_out]; num_nodes_in];
        let cost_gradient_biases = vec![0.0; num_nodes_out];

        let weights = Self::initialize_random_weights(num_nodes_in, num_nodes_out);
        let biases = vec![0.0; num_nodes_out];

        Layer {
            num_nodes_in,
            num_nodes_out,
            cost_gradient_weights,
            cost_gradient_biases,
            weights,
            biases,
        }
    }

    // Update the weights and biases based on the cost gradients (gradient descent).
    pub fn apply_gradients(&mut self, learn_rate: f32) {
        for node_out in 0..self.num_nodes_out {
            self.biases[node_out] -= self.cost_gradient_biases[node_out] * learn_rate;

            for node_in in 0..self.num_nodes_in {
                self.weights[node_in][node_out] -= self.cost_gradient_weights[node_in][node_out] * learn_rate;
            }
        }
    }

    // Calculated the output of the layer.
    pub fn calculate_outputs(&self, inputs: Vec<f32>) -> Vec<f32> {
        let mut activations = vec![0.0; self.num_nodes_out];

        for node_out in 0..self.num_nodes_out {
            let mut weighted_input = self.biases[node_out];
            for node_in in 0..self.num_nodes_in {
                weighted_input += inputs[node_in] * self.weights[node_in][node_out];
            }
            let activation = self.activation_function(weighted_input);
            activations[node_out] = activation;
        }

        return activations;
    }

    pub fn node_cost(&self, output_activation: f32, expected_output: f32) -> f32 {
        let error = output_activation - expected_output;

        error * error
    }

    pub fn activation_function(&self, weighted_input: f32) -> f32 {
        self._sigmoid(weighted_input)
    }

    fn initialize_random_weights(num_nodes_in: usize, num_nodes_out: usize) -> Vec<Vec<f32>> {
        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(num_nodes_in);

        let min = -WEIGHT_BOUNDRY;
        let max = WEIGHT_BOUNDRY;

        for _ in 0..num_nodes_in {
            let mut current_weights: Vec<f32> = vec![];
            for _ in 0..num_nodes_out {
                let random_weight = min + snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32 * (max - min);
                current_weights.push(random_weight);
            }
            weights.push(current_weights);
        }

        weights
    }

    // Step
    fn _step(&self, weighted_input: f32) -> f32 {
        if weighted_input > 0.0 { 1.0 } else { 0.0 }
    }

    // Sigmoid
    fn _sigmoid(&self, weighted_input: f32) -> f32 {
        1.0 / (1.0 + (-weighted_input).exp())
    }

    // Hyperbolic tangent
    fn _hyperbolic_tangent(&self, weighted_input: f32) -> f32 {
        let e2w = (2.0 * weighted_input).exp();
        (e2w - 1.0) / (e2w + 1.0)
    }

    // SiLU
    fn _silu(&self, weighted_input: f32) -> f32 {
        weighted_input / (1.0 + (-weighted_input).exp())
    }

    // ReLU
    fn _relu(&self, weighted_input: f32) -> f32 {
        weighted_input.max(0.0)
    }
}

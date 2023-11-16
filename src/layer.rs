use snippets;

const WEIGHT_BOUNDRY: f32 = 1.0;
const BIAS_BOUNDRY: f32 = 1.0;

pub struct Layer {
    num_nodes_in: usize,
    num_nodes_out: usize,
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
}

impl Layer {

    // Create the layer.
    pub fn new(num_nodes_in: usize, num_nodes_out: usize) -> Self {
        let mut weights: Vec<Vec<f32>> = Vec::with_capacity(num_nodes_in);
        let min_weight = -WEIGHT_BOUNDRY;
        let max_weight = WEIGHT_BOUNDRY;
        for _ in 0..num_nodes_in {
            let mut current_weights: Vec<f32> = vec![];
            for _ in 0..num_nodes_out {
                current_weights.push(min_weight + snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32 * (max_weight - min_weight));
            }
            weights.push(current_weights);
        }

        let mut biases: Vec<f32> = Vec::with_capacity(num_nodes_out);
        let min_bias = -BIAS_BOUNDRY;
        let max_bias = BIAS_BOUNDRY;
        for _ in 0..num_nodes_out {
            biases.push(min_bias + snippets::random_numbers().next().unwrap() as f32 / std::u64::MAX as f32 * (max_bias - min_bias));
        }

        Layer {
            num_nodes_in,
            num_nodes_out,
            weights,
            biases,
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

    pub fn activation_function(&self, weighted_input: f32) -> f32 {
        self._sigmoid(weighted_input)
    }

    pub fn node_cost(&self, output_activation: f32, expected_output: f32) -> f32 {
        let error = output_activation - expected_output;

        error * error
    }

    // Step
    fn _step(&self, weighted_input: f32) -> f32 {
        if weighted_input > 0.0 { 1.0 } else { 0.0 }
    }

    // Sigmoid
    fn _sigmoid(&self, weighted_input: f32) -> f32 {
        1.0 / (1.0 + -weighted_input.exp())
    }

    // Hyperbolic tangent
    fn _hyperbolic_tangent(&self, weighted_input: f32) -> f32 {
        let e2w = (2.0 * weighted_input).exp();
        (e2w - 1.0) / (e2w + 1.0)
    }

    // SiLU
    fn _silu(&self, weighted_input: f32) -> f32 {
        weighted_input / (1.0 + -weighted_input.exp())
    }

    // ReLU
    fn _relu(&self, weighted_input: f32) -> f32 {
        weighted_input.max(0.0)
    }
}

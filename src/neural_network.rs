use crate::activation::{Activation, Activations, ActivationType};
use crate::layer::Layer;
use crate::data_point::DataPoint;

pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub activation: Box<dyn Activation>,
}

impl NeuralNetwork {
    // Create new neural network.
    pub fn new(layer_sizes: Vec<usize>, activation_type: &ActivationType) -> Self {
        // Do not include the input layer in the layers: -1
        let layer_count = layer_sizes.len() - 1;
        let mut layers: Vec<Layer> = Vec::with_capacity(layer_count);

        for i in 0..layer_count {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }

        let activation = Activations::get_activation(activation_type);

        NeuralNetwork {
            layers,
            activation,
        }
    }

    pub fn learn(&mut self, training_data: &Vec<DataPoint>, learn_rate: f32, h: f32) {
        let original_cost = self.cost(training_data);

        // Clone layers to use for iterating with a immutable reference.
        let layers = self.layers.clone();

        for layer_index in 0..layers.len() {
            let layer = &layers[layer_index];

            // Calculate the cost gradient for the current weights.
            for node_in in 0..layer.num_nodes_in {
                for node_out in 0..layer.num_nodes_out {
                    self.layers[layer_index].weights[node_in][node_out] += h;
                    let delta_cost = self.cost(training_data) - original_cost;
                    self.layers[layer_index].weights[node_in][node_out] -= h;
                    self.layers[layer_index].cost_gradient_weights[node_in][node_out] = delta_cost / h;
                }
            }

            // Calculate the cost gradient for the current biases.
            for node_out in 0..layer.num_nodes_out {
                self.layers[layer_index].biases[node_out] += h;
                let delta_cost = self.cost(training_data) - original_cost;
                self.layers[layer_index].biases[node_out] -= h;
                self.layers[layer_index].cost_gradient_biases[node_out] = delta_cost / h;
            }
        }

        self.apply_all_gradients(learn_rate);
    }

    fn apply_all_gradients(&mut self, learn_rate: f32) {
        for layer in &mut self.layers {
            layer.apply_gradients(learn_rate);
        }
    }

    // Run the input values through the network to calculate the output values.
    pub fn calculate_outputs(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs = inputs;

        for layer_index in 0..self.layers.len() {
            self.layers[layer_index].calculate_outputs(outputs, &self.activation);

            outputs = self.layers[layer_index].activations.clone();
        }

        outputs
    }

    // Run the inputs through the network and calculate which output node has the highest value.
    pub fn classify(&mut self, inputs: Vec<f32>) -> Option<usize> {
        let outputs = self.calculate_outputs(inputs);

        outputs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
    }

    pub fn cost(&mut self, data: &Vec<DataPoint>) -> f32 {
        let mut total_cost: f32 = 0.0;
        let data_len = data.len() as f32;

        for data_point in data {
            total_cost += self.cost_single(data_point);
        }

        total_cost / data_len
    }

    fn cost_single(&mut self, data_point: &DataPoint) -> f32 {
        let outputs = self.calculate_outputs(data_point.inputs.to_vec());
        let output_layer = &self.layers.last().unwrap();
        let mut cost: f32 = 0.0;

        for node_out in 0..outputs.len() {
            let current_node_cost = output_layer.node_cost(outputs[node_out], data_point.expected_outputs[node_out]);
            cost += current_node_cost;
        }

        cost
    }
}
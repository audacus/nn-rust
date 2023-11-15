use crate::layer::Layer;
use crate::data_point::DataPoint;

pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {

    // Create new neural network.
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        // Do not include the input layer in the layers: -1
        let layer_count = layer_sizes.len() - 1;
        let mut layers: Vec<Layer> = Vec::with_capacity(layer_count);

        for i in 0..layer_count {
            layers.push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }

        NeuralNetwork { layers }
    }

    // Run the input values through the network to calculate the output values.
    pub fn calculate_outputs(&self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut outputs = inputs.clone();

        // Pass the values through every layer.
        for layer in &self.layers {
            outputs = layer.calculate_outputs(outputs);
        }

        outputs
    }

    // Run the inputs through the newtwork and calculate which output node has the highest value.
    pub fn classify(&self, inputs: &Vec<f32>) -> Option<usize> {
        let outputs = self.calculate_outputs(inputs);

        outputs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
    }

    pub fn cost(&self, data: &Vec<DataPoint>) -> f32 {
        let mut total_cost: f32 = 0.0;
        let data_len = data.len() as f32;

        for data_point in data {
            total_cost += self.cost_single(data_point);
        }

        total_cost / data_len
    }

    fn cost_single(&self, data_point: &DataPoint) -> f32 {
        let outputs = self.calculate_outputs(&data_point.inputs);
        let output_layer = &self.layers[self.layers.len() - 1];
        let mut cost: f32 = 0.0;

        for node_out in 0..outputs.len() {
            cost += output_layer.node_cost(outputs[node_out], data_point.expected_outputs[node_out]);
        }

        cost
    }
}
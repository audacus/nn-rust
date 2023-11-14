use crate::layer::Layer;

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
    pub fn calculate_outputs(&self, mut inputs: Vec<f32>) -> Vec<f32> {
        for layer in &self.layers {
            inputs = layer.calculate_outputs(inputs);
        }

        inputs
    }

    // Run the inputs through the newtwork and calculate which output node has the highest value.
    pub fn classify(&self, inputs: Vec<f32>) -> Option<usize> {
        let outputs = self.calculate_outputs(inputs);

        outputs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index)
    }
}
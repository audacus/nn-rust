pub struct DataPoint {
    pub inputs: Vec<f32>,
    pub expected_outputs: Vec<f32>,
    pub label: usize,
}

impl DataPoint {
    pub fn new(inputs: Vec<f32>, label: usize, num_labels: usize) -> Self {
        DataPoint {
            inputs,
            label,
            expected_outputs: Self::create_one_hot(label, num_labels),
        }
    }

    fn create_one_hot(index: usize, num: usize) -> Vec<f32> {
        let mut one_hot = vec![0.0; num];
        one_hot[index] = 1.0;

        one_hot
    }
}
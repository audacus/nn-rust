pub struct GradientDescent {
    pub input_value: f32,
    pub learn_rate: f32,
    pub h: f32,
    pub past_values: Vec<f32>,
}

impl GradientDescent {
    pub const LEARN_RATE_STEP: f32 = 0.05;
    pub const H_FACTOR: f32 = 10.0;

    pub fn new(random_value: f32) -> Self {
        GradientDescent {
            input_value: random_value,
            learn_rate: 2.3,
            h: 0.0001,
            past_values: vec![],
        }
    }

    pub fn function(x: f32) -> f32 {
        0.2 * x.powf(4.0) + 0.1 * x.powf(3.0) - x.powf(2.0) + 2.0
    }

    // Run one iteration of the gradient descent algorithm.
    pub fn learn(&mut self) {
        println!("gradient descent learn!");

        self.past_values.push(self.input_value);

        let delta_output = Self::function(self.input_value + self.h) - Self::function(self.input_value);
        let slope = delta_output / self.h;

        self.input_value -= slope * self.learn_rate;
    }
}
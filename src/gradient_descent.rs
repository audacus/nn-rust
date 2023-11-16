pub struct GradientDescent {
    pub input_value: f32,
    pub learn_rate: f32,
}

impl GradientDescent {

    pub fn new(random_value: f32) -> Self {
        GradientDescent {
            input_value: random_value,
            learn_rate: 0.0,
        }
    }

    pub fn function(x: f32) -> f32 {
        0.2 * x.powf(4.0) + 0.1 * x.powf(3.0) - x.powf(2.0) + 2.0
    }

    // Run one iteration of the grdient descent algorithm.
    pub fn learn (&mut self) {
        println!("learn!");
        // Approximate the slope of the function using the finite difference method.
        let h: f32 = 0.00001;

        let delta_output = Self::function(self.input_value + h) - Self::function(self.input_value);
        let slope = delta_output / h;

        self.input_value -= slope * self.learn_rate;
    }
}
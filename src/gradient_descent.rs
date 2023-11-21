pub struct GradientDescent {
    pub input_value: f32,
    pub learn_rate: f32,
    // Approximate the slope of the function using the finite difference method.
    pub h: f32,
}

impl GradientDescent {
    pub const LEARN_RATE_STEP: f32 = 0.25;
    pub const H_FACTOR: f32 = 10.0;

    pub fn new(random_value: f32) -> Self {
        GradientDescent {
            input_value: random_value,
            learn_rate: 2.25,
            h: 0.0001
        }
    }

    pub fn function(x: f32) -> f32 {
        0.2 * x.powf(4.0) + 0.1 * x.powf(3.0) - x.powf(2.0) + 2.0
    }

    // Run one iteration of the grdient descent algorithm.
    pub fn learn (&mut self) {
        println!("gradient descent learn!");

        let delta_output = Self::function(self.input_value + self.h) - Self::function(self.input_value);
        let slope = delta_output / self.h;

        self.input_value -= slope * self.learn_rate;
    }
}
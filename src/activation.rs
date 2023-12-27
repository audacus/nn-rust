pub trait Activation {
    fn activate(&self, input: f32) -> f32;
    fn derivative(&self, input: f32) -> f32;
}

struct Sigmoid;

struct TanH;

struct ReLU;

struct SiLU;

impl Activation for Sigmoid {
    fn activate(&self, input: f32) -> f32 {
        1.0 / (1.0 + (-input).exp())
    }

    fn derivative(&self, input: f32) -> f32 {
        let activate = self.activate(input);

        activate * (1.0 - activate)
    }
}

impl Activation for TanH {
    fn activate(&self, input: f32) -> f32 {
        let e2 = (2.0 * input).exp();

        (e2 - 1.0) / (e2 + 1.0)
    }

    fn derivative(&self, input: f32) -> f32 {
        let t = self.activate(input);

        1.0 - t * t
    }
}

impl Activation for ReLU {
    fn activate(&self, input: f32) -> f32 {
        input.max(0.0)
    }

    fn derivative(&self, input: f32) -> f32 {
        if input > 0.0 { 1.0 } else { 0.0 }
    }
}

impl Activation for SiLU {
    fn activate(&self, input: f32) -> f32 {
        input / (1.0 + (-input).exp())
    }

    fn derivative(&self, input: f32) -> f32 {
        let sig = 1.0 / (1.0 + (-input).exp());

        input * sig * (1.0 - sig) + sig
    }
}

#[allow(dead_code)]
pub enum ActivationType {
    SIGMOID,
    TANH,
    RELU,
    SILU,
}

pub struct Activations;

impl Activations {
    pub fn get_activation(activation_type: &ActivationType) -> Box<dyn Activation> {
        match activation_type {
            ActivationType::SIGMOID => Box::new(Sigmoid {}),
            ActivationType::TANH => Box::new(TanH {}),
            ActivationType::RELU => Box::new(ReLU {}),
            ActivationType::SILU => Box::new(SiLU {}),
        }
    }
}

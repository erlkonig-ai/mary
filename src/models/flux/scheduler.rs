use burn::prelude::*;

/// FlowMatchEulerDiscreteScheduler — pure math, no learned weights.
///
/// Implements the Euler method for flow matching ODE sampling.
pub struct FlowMatchEulerDiscreteScheduler {
    pub num_train_timesteps: usize,
    pub sigmas: Vec<f32>,
    pub timesteps: Vec<f32>,
    pub step_index: usize,
}

impl FlowMatchEulerDiscreteScheduler {
    /// Create a new scheduler and set timesteps with dynamic shifting.
    pub fn new(num_inference_steps: usize, image_seq_len: usize) -> Self {
        let num_train_timesteps: usize = 1000;

        // Compute mu for dynamic shifting
        let mu = compute_empirical_mu(image_seq_len, num_inference_steps);

        // sigmas = linspace(1.0, 1/N, N)
        let mut sigmas: Vec<f32> = (0..num_inference_steps)
            .map(|i| {
                let sigma_max = 1.0_f64;
                let sigma_min = 1.0 / num_inference_steps as f64;
                sigma_max
                    + (sigma_min - sigma_max) * (i as f64) / (num_inference_steps as f64 - 1.0)
            })
            .map(|s| s as f32)
            .collect();

        // Apply dynamic (exponential) time shift: exp(mu) / (exp(mu) + (1/t - 1))
        let exp_mu = mu.exp();
        sigmas = sigmas
            .iter()
            .map(|&t| {
                let t = t as f64;
                let shifted = exp_mu / (exp_mu + (1.0 / t - 1.0));
                shifted as f32
            })
            .collect();

        // Compute timesteps = sigmas * num_train_timesteps
        let timesteps: Vec<f32> = sigmas
            .iter()
            .map(|&s| s * num_train_timesteps as f32)
            .collect();

        // Append terminal sigma = 0
        sigmas.push(0.0);

        Self {
            num_train_timesteps,
            sigmas,
            timesteps,
            step_index: 0,
        }
    }

    /// Reset step index for a new denoising run.
    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    /// Get the current timestep value.
    pub fn current_timestep(&self) -> f32 {
        self.timesteps[self.step_index]
    }

    /// Perform one Euler step.
    /// model_output: predicted velocity field from the transformer
    /// sample: current noisy latents
    /// Returns: updated latents (prev_sample)
    pub fn step<B: Backend>(
        &mut self,
        model_output: Tensor<B, 3>,
        sample: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;

        // Euler step: prev_sample = sample + dt * model_output
        let prev_sample = sample + model_output.mul_scalar(dt);

        self.step_index += 1;
        prev_sample
    }

    /// Number of inference steps.
    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }
}

/// Compute the empirical mu parameter for dynamic time shifting.
/// Based on a linear fit of optimal mu vs image sequence length and number of steps.
pub fn compute_empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let a1: f64 = 8.73809524e-05;
    let b1: f64 = 1.89833333;
    let a2: f64 = 0.00016927;
    let b2: f64 = 0.45666666;

    let seq = image_seq_len as f64;

    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }

    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;

    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

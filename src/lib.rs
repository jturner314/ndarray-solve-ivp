pub mod rk;

use ndarray::prelude::*;
use std::error::Error;

pub trait OdeIntegrate {
    /// Returns the number of elements in the state.
    fn len(&self) -> usize;
    /// Perform one step (adaptive step size).
    fn step(&mut self) -> Result<(), Box<dyn Error>>;
    /// Current time.
    fn time(&self) -> f64;
    /// The ending time.
    fn time_bound(&self) -> f64;
    /// Current state.
    fn state(&self) -> ArrayView1<'_, f64>;
    /// Returns `true` if the integration has reached `time_bound`.
    fn finished(&self) -> bool {
        self.time() == self.time_bound()
    }
    /// Integrate until reaching `time_bound`.
    fn run_to_bound(&mut self) -> Result<(), Box<dyn Error>> {
        while !self.finished() {
            self.step()?;
        }
        Ok(())
    }
}

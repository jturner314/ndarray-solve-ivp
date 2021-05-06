#[macro_use]
extern crate ndarray;
extern crate ndarray_solve_ivp;

use ndarray::prelude::*;

use ndarray_solve_ivp::rk::{RungeKutta, RK45};
use ndarray_solve_ivp::OdeIntegrate;

fn fun_rational(t: f64, y: ArrayView1<f64>, mut dy: ArrayViewMut1<f64>) {
    dy[0] = y[1] / t;
    dy[1] = y[1] * (y[0] + 2. * y[1] - 1.) / (t * (y[0] - 1.));
}

fn sol_rational(t: f64) -> Array1<f64> {
    array![t / (t + 10.), 10. * t / (t + 10.).powi(2)]
}

fn compute_error(
    y: ArrayView1<f64>,
    y_true: ArrayView1<f64>,
    rtol: ArrayView1<f64>,
    atol: ArrayView1<f64>,
) -> Array1<f64> {
    let e = (&y - &y_true) / (y_true.mapv(f64::abs) * &rtol + &atol);
    (&e * &e / e.len() as f64).mapv(f64::sqrt)
}

#[test]
fn integration() {
    let t0 = 5.;
    let y0 = array![1. / 3., 2. / 9.];
    let t_bound = 9.;
    let max_step = ::std::f64::INFINITY;
    let rtol = Array1::from_elem(y0.len(), 1e-3);
    let atol = Array1::from_elem(y0.len(), 1e-6);
    let mut solver = RungeKutta::<_, RK45>::new(
        fun_rational,
        t0,
        y0,
        t_bound,
        max_step,
        rtol.clone(),
        atol.clone(),
    )
    .unwrap();
    solver.run_to_bound().unwrap();
    assert!(compute_error(
        solver.state(),
        sol_rational(t_bound).view(),
        rtol.view(),
        atol.view()
    )
    .iter()
    .all(|&e| e < 5.));
}

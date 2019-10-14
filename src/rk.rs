//! Runge–Kutta solvers.

use ndarray::{FoldWhile, Zip};
use ndarray::prelude::*;
use std::error::Error;
use std::marker::PhantomData;

use OdeIntegrate;

/// Multiply steps computed from asymptotic behaviour of errors by this.
const SAFETY: f64 = 0.9;
/// Minimum allowed decrease in a step size.
const MIN_FACTOR: f64 = 0.2;
/// Maximum allowed increase in a step size.
const MAX_FACTOR: f64 = 10.;

/// Computes RMS norm of scaled values.
fn norm(x: ArrayView1<f64>, scale: ArrayView1<f64>) -> f64 {
    debug_assert_eq!(x.len(), scale.len());
    (Zip::from(x)
        .and(scale)
        .fold_while(0., |acc, &x, &scale| {
            let scaled = x / scale;
            FoldWhile::Continue(acc + scaled * scaled)
        })
        .into_inner() / x.len() as f64)
        .sqrt()
}

/// Empirically select a good initial step.
///
/// The algorithm is described in (ref 1).
///
/// # Parameters
///
/// * fun: Right-hand side of the system.
/// * t0: Initial value of the independent variable.
/// * y0: Initial value of the dependent variable.
/// * f0: Initial value of the derivative, i.e. the result of calling `fun`
///   with `t0` and `y0`.
/// * direction: Integration direction.
/// * order: Method order.
/// * rtol: Desired relative tolerance.
/// * atol: Desired absolute tolerance.
///
/// # Returns
///
/// Absolute value of the suggested initial step.
///
/// # References
///
/// 1. E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
///    Equations I: Nonstiff Problems", Sec. II.4.
fn select_initial_step<F>(
    fun: &mut F,
    t0: f64,
    y0: ArrayView1<f64>,
    f0: ArrayView1<f64>,
    direction: f64,
    order: usize,
    rtol: ArrayView1<f64>,
    atol: ArrayView1<f64>,
) -> f64
where
    F: FnMut(f64, ArrayView1<f64>, ArrayViewMut1<f64>),
{
    if y0.is_empty() {
        return ::std::f64::INFINITY;
    }

    let scale = y0.mapv(f64::abs) * rtol + atol;
    let d0 = norm(y0, scale.view());
    let d1 = norm(f0, scale.view());
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    let y1 = h0 * direction * &f0 + y0;
    let mut f1 = Array1::zeros(y0.len());
    fun(t0 + h0 * direction, y1.view(), f1.view_mut());
    let d2 = norm((f1 - f0).view(), scale.view()) / h0;

    let h1 = if d1 <= 1e-15 && d2 <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / d1.max(d2)).powf(1. / (order as f64 + 1.))
    };

    return (100. * h0).min(h1);
}

/// Runge–Kutta ODE IVP solver.
pub struct RungeKutta<F, O>
where
    F: FnMut(f64, ArrayView1<f64>, ArrayViewMut1<f64>),
    O: RKMethod,
{
    fun: F,
    order: PhantomData<O>,
    /// Current time.
    t: f64,
    /// Current state.
    y: Array1<f64>,
    /// Previous time, or `None` if there haven't been any steps.
    t_old: Option<f64>,
    /// Previous state, or `None` if there haven't been any steps.
    y_old: Option<Array1<f64>>,
    /// Boundary time.
    t_bound: f64,
    /// Integration direction: +1 or -1.
    direction: f64,
    /// Maximum step size.
    ///
    /// Can be NAN or INFINITY to indicate that the step is not bounded and is
    /// determined solely by the solver.
    max_step: f64,
    /// Initial step size for next `.step()`.
    h_abs: f64,
    /// Relative tolerance.
    rtol: Array1<f64>,
    /// Absolute tolerance.
    atol: Array1<f64>,
    /// Storage array for Runge Kutta stages, shape `O::NUM_STAGES + 1, self.len()`.
    k: Array2<f64>,
}

quick_error! {
    #[derive(Debug)]
    pub enum CreateRungeKuttaError {
        TimeBoundNotFinite {
            description("t_bound is not finite")
            display(x) -> ("{}", x.description())
        }
        MaxStepZeroOrNeg {
            description("max_step is zero or negative")
            display(x) -> ("{}", x.description())
        }
        UnequalLengths {
            description("array arguments have unequal lengths")
            display(x) -> ("{}", x.description())
        }
        /// The relative tolerance was too small.
        ///
        /// It must be at least `100. * ::std::f64::EPSILON`.
        TooSmallRelTol {
            description("rtol is too small")
            display(x) -> ("{}", x.description())
        }
    }
}

struct StepOutput {
    /// Solution at `t + h` computed with higher accuracy.
    y_new: Array1<f64>,
    /// Error estimate of less accurate method.
    error: Array1<f64>,
}

impl<F, O> RungeKutta<F, O>
where
    F: FnMut(f64, ArrayView1<f64>, ArrayViewMut1<f64>),
    O: RKMethod,
{
    /// Creates a new `RungeKutta` solver.
    ///
    /// # Parameters
    ///
    /// * `fun`: Right-hand side of the system, where calling `fun(t, y,
    ///   deriv_y)` should fill in `deriv_y` with the derivative of `y` at time
    ///   `t`.
    ///
    /// * `t0`: Initial value of the independent variable.
    ///
    /// * `y0`: Initial values of the dependent variable.
    ///
    /// * `t_bound`: Boundary time — the integration won't continue beyond
    ///   it. It also determines the direction of the integration.
    ///
    /// * `max_step`: Maximum allowed step size.
    ///
    /// * `rtol`, `atol`: Relative and absolute tolerances. The solver keeps
    ///   the local error estimates less than `atol + rtol * abs(y)`. Here
    ///   `rtol` controls a relative accuracy (number of correct digits). But
    ///   if a component of `y` is approximately below `atol` then the error
    ///   only needs to fall within the same `atol` threshold, and the number
    ///   of correct digits is not guaranteed. If components of y have
    ///   different scales, it might be beneficial to set different `atol`
    ///   values for different components.
    pub fn new(
        mut fun: F,
        t0: f64,
        y0: Array1<f64>,
        t_bound: f64,
        max_step: f64,
        rtol: Array1<f64>,
        atol: Array1<f64>,
    ) -> Result<RungeKutta<F, O>, CreateRungeKuttaError> {
        if !t_bound.is_finite() {
            return Err(CreateRungeKuttaError::TimeBoundNotFinite);
        }
        if max_step <= 0. {
            return Err(CreateRungeKuttaError::MaxStepZeroOrNeg);
        }
        if y0.len() != rtol.len() || y0.len() != atol.len() {
            return Err(CreateRungeKuttaError::UnequalLengths);
        }
        if rtol.fold(false, |acc, &tol| acc && tol < 100. * ::std::f64::EPSILON) {
            return Err(CreateRungeKuttaError::TooSmallRelTol);
        }

        let direction = if t_bound < t0 { -1. } else { 1. };

        // Initialize state derivative (stored in last row of k).
        let mut k = Array2::zeros((O::NUM_STAGES + 1, y0.len()));
        fun(t0, y0.view(), k.slice_mut(s![-1, ..]));

        // Determine initial step size.
        let h_abs = select_initial_step(
            &mut fun,
            t0,
            y0.view(),
            k.slice(s![-1, ..]),
            direction,
            O::ORDER,
            rtol.view(),
            atol.view(),
        );

        Ok(RungeKutta {
            fun,
            order: PhantomData,
            t: t0,
            y: y0,
            t_old: None,
            y_old: None,
            t_bound,
            direction,
            max_step,
            h_abs,
            rtol,
            atol,
            k,
        })
    }

    /// Current state derivative.
    pub fn state_deriv(&self) -> ArrayView1<f64> {
        self.k.slice(s![-1, ..])
    }

    /// Size of last successful step or `None` if no steps were made yet.
    pub fn step_size(&self) -> Option<f64> {
        self.t_old.map(|t_old| (self.t - t_old).abs())
    }

    /// Perform a single Runge–Kutta step.
    ///
    /// This function computes a prediction of an explicit Runge–Kutta method and
    /// also estimates the error of a less accurate method.
    /// Notation for Butcher tableau is as in (ref 1).
    ///
    /// # References
    ///
    /// 1. E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
    ///    Equations I: Nonstiff Problems", Sec. II.4.
    fn step_by(&mut self, h: f64) -> StepOutput {
        // TODO: remove this to_owned
        let f = self.state_deriv().to_owned();
        self.k.slice_mut(s![0, ..]).assign(&f);
        for (s, (a, c)) in O::a().iter().zip(O::c()).enumerate() {
            let dy = self.k.slice(s![..s + 1, ..]).t().dot(a) * h;
            (self.fun)(
                self.t + c * h,
                (dy + &self.y).view(),
                self.k.slice_mut(s![s + 1, ..]),
            );
        }

        let y_new = h * self.k.slice::<Ix2>(s![..-1, ..]).t().dot(&O::b()) + &self.y;
        (self.fun)(self.t + h, y_new.view(), self.k.slice_mut(s![-1, ..]));

        let error = self.k.t().dot(&O::e()) * h;

        StepOutput { y_new, error }
    }
}

quick_error! {
    #[derive(Debug)]
    pub enum RungeKuttaStepError {
        TooSmallStep(required: f64, allowable: f64) {
            description("required step size is too small")
            display(x) -> ("required step size {} is smaller than min allowable step size {}", required, allowable)
        }
    }
}

/// Computes the next representable floating-point value following `x` in the
/// direction of `y`.
///
/// Special cases:
///
/// * If `x` equals `y`, then `y` is returned.
/// * If `x` or `y` is NAN, a NAN is returned.
///
/// There is no special handling for overflow of finite values to ±∞ or
/// subnormals.
fn next_after(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        ::std::f64::NAN
    } else if x == y {
        y
    } else if x == 0. {
        if y < 0. {
            -f64::from_bits(1)
        } else {
            f64::from_bits(1)
        }
    } else if (y > x) == (x > 0.) {
        f64::from_bits(x.to_bits().wrapping_add(1))
    } else {
        f64::from_bits(x.to_bits().wrapping_sub(1))
    }
}

impl<F, O> OdeIntegrate for RungeKutta<F, O>
where
    F: FnMut(f64, ArrayView1<f64>, ArrayViewMut1<f64>),
    O: RKMethod,
{
    fn len(&self) -> usize {
        self.y.len()
    }

    fn step(&mut self) -> Result<(), Box<Error>> {
        let min_step =
            10. * (next_after(self.t, self.direction * ::std::f64::INFINITY) - self.t).abs();

        let mut h_abs = self.h_abs.min(self.max_step).max(min_step);

        let mut scale = Array1::zeros(self.y.len());
        loop {
            // Update `h_abs` and calcualte `t_new` and `h`.
            if h_abs < min_step {
                return Err(Box::new(RungeKuttaStepError::TooSmallStep(h_abs, min_step)));
            }
            let t_new = if h_abs >= (self.t_bound - self.t).abs() {
                self.t_bound
            } else {
                self.t + h_abs * self.direction
            };
            let h = t_new - self.t;
            h_abs = h.abs();

            // Perform step and calculate error norm.
            let StepOutput { y_new, error } = self.step_by(h);
            azip!((
                scale in &mut scale,
                &y in &self.y,
                &y_new in &y_new,
                &atol in &self.atol,
                &rtol in &self.rtol,
            ) {
                *scale = atol + y.abs().max(y_new.abs()) * rtol;
            });
            let error_norm = norm(error.view(), scale.view());

            // Accept or reject step based on error norm.
            if error_norm < 1. {
                self.t_old = Some(self.t);
                self.t = t_new;
                self.y_old = Some(::std::mem::replace(&mut self.y, y_new));
                self.h_abs = h_abs
                    * MAX_FACTOR
                        .min((SAFETY * error_norm.powf(-1. / (O::ORDER as f64 + 1.))).max(1.));
                return Ok(());
            } else {
                h_abs *= MIN_FACTOR.max(SAFETY * error_norm.powf(-1. / (O::ORDER as f64 + 1.)));
            }
        }
    }

    fn time(&self) -> f64 {
        self.t
    }

    fn time_bound(&self) -> f64 {
        self.t_bound
    }
    /// Current state.
    fn state(&self) -> ArrayView1<f64> {
        self.y.view()
    }
}

pub trait RKMethod {
    /// Order of the method.
    const ORDER: usize;

    /// Number of stages in the method.
    const NUM_STAGES: usize;

    /// Coefficients for incrementing time for consecutive RK stages, length
    /// `NUM_STAGES - 1`.
    ///
    /// The value for the first stage is always zero, so it is not included.
    fn c() -> ArrayView1<'static, f64>;

    /// Coefficients for combining previous RK stages to compute the next
    /// stage, length `NUM_STAGES - 1`.
    ///
    /// For explicit methods the coefficients above the main diagonal are
    /// zeros, so `a` is stored as a list of arrays of increasing lengths. The
    /// first stage is always just `f`, thus no coefficients for it are
    /// required.
    fn a() -> &'static [ArrayView1<'static, f64>];

    /// Coefficients for combining RK stages for computing the final
    /// prediction, length `NUM_STAGES`.
    fn b() -> ArrayView1<'static, f64>;

    /// Coefficients for estimating the error of a less accurate method, length
    /// `NUM_STAGES + 1`.
    ///
    /// They are computed as the difference between `b`'s in an extended
    /// tableau.
    fn e() -> ArrayView1<'static, f64>;

    /// Polynomial coefficients for dense output.
    fn p() -> ArrayView2<'static, f64>;
}

/// Explicit Runge–Kutta method of order 3(2).
///
/// The Bogacki-Shamping pair of formulas is used (ref 1). The error is
/// controlled assuming 2nd order accuracy, but steps are taken using a 3rd
/// oder accurate formula (local extrapolation is done). A cubic Hermit
/// polynomial is used for the dense output.
///
/// # References
///
/// 1. P. Bogacki, L.F. Shampine, "A 3(2) Pair of Runge-Kutta Formulas",
///    Appl. Math. Lett. Vol. 2, No. 4. pp. 321-325, 1989.
pub struct RK23;

impl RKMethod for RK23 {
    const ORDER: usize = 2;

    const NUM_STAGES: usize = 3;

    fn c() -> ArrayView1<'static, f64> {
        aview1(&[1./2., 3./4.])
    }

    fn a() -> &'static [ArrayView1<'static, f64>] {
        lazy_static! {
            static ref A: [ArrayView1<'static, f64>; 3 - 1] = [
                aview1(&[1./2.]),
                aview1(&[0., 3./4.]),
            ];
        }
        &*A
    }

    fn b() -> ArrayView1<'static, f64> {
        aview1(&[2./9., 1./3., 4./9.])
    }

    fn e() -> ArrayView1<'static, f64> {
        aview1(&[5./72., -1./12., -1./9., 1./8.])
    }

    fn p() -> ArrayView2<'static, f64> {
        aview2(&[
            [1., -4./3., 5./9.],
            [0., 1., -2./3.],
            [0., 4./3., -8./9.],
            [0., -1., 1.],
        ])
    }
}

/// Explicit Runge–Kutta method of order 5(4).
///
/// The Dormand-Prince pair of formulas is used (ref 1). The error is
/// controlled assuming 4th order accuracy, but steps are taken using a 5th
/// oder accurate formula (local extrapolation is done). A quartic
/// interpolation polynomial is used for the dense output (ref 2).
///
/// # References
///
/// 1. J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta
///    formulae", Journal of Computational and Applied Mathematics, Vol. 6, No.
///    1, pp. 19-26, 1980.
///
/// 2. L. W. Shampine, "Some Practical Runge-Kutta Formulas", Mathematics of
///    Computation,, Vol. 46, No. 173, pp. 135-150, 1986.
pub struct RK45;

impl RKMethod for RK45 {
    const ORDER: usize = 4;

    const NUM_STAGES: usize = 6;

    fn c() -> ArrayView1<'static, f64> {
        aview1(&[1./5., 3./10., 4./5., 8./9., 1.])
    }

    fn a() -> &'static [ArrayView1<'static, f64>] {
        lazy_static! {
            static ref A: [ArrayView1<'static, f64>; 6 - 1] = [
                aview1(&[1./5.]),
                aview1(&[3./40., 9./40.]),
                aview1(&[44./45., -56./15., 32./9.]),
                aview1(&[19372./6561., -25360./2187., 64448./6561., -212./729.]),
                aview1(&[9017./3168., -355./33., 46732./5247., 49./176., -5103./18656.]),
            ];
        }
        &*A
    }

    fn b() -> ArrayView1<'static, f64> {
        aview1(&[35./384., 0., 500./1113., 125./192., -2187./6784., 11./84.])
    }

    fn e() -> ArrayView1<'static, f64> {
        aview1(&[-71./57600., 0., 71./16695., -71./1920., 17253./339200., -22./525., 1./40.])
    }

    fn p() -> ArrayView2<'static, f64> {
        // Corresponds to the optimum value of c_6 from (ref 2).
        aview2(&[
            [1., -8048581381./2820520608., 8663915743./2820520608., -12715105075./11282082432.],
            [0., 0., 0., 0.],
            [0., 131558114200./32700410799., -68118460800./10900136933.,
             87487479700./32700410799.],
            [0., -1754552775./470086768., 14199869525./1410260304., -10690763975./1880347072.],
            [0., 127303824393./49829197408., -318862633887./49829197408.,
             701980252875. / 199316789632.],
            [0., -282668133./205662961., 2019193451./616988883., -1453857185./822651844.],
            [0., 40617522./29380423., -110615467./29380423., 69997945./29380423.]
        ])
    }
}

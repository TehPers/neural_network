use ndarray::{Array1, NdFloat};

/// An activation function.
pub trait ActivationFn<Scalar = f64> {
    /// Applies the activation function to the input.
    fn activate(&self, input: Array1<Scalar>) -> Array1<Scalar>;

    /// Derivative of the activation function with respect to the inputs. In other words, if `y` is
    /// the output, `x` is the input, and `a` is the activation function and `y = a(x)`, then this
    /// function returns `dy/dx`, which is the same as `da(x)/dx`.
    fn activate_prime(&self, input: Array1<Scalar>) -> Array1<Scalar>;
}

/// The sigmoid activation function.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sigmoid;

// type Scalar = f64;
impl<Scalar> ActivationFn<Scalar> for Sigmoid
where
    Scalar: NdFloat,
{
    fn activate(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.mapv_into(|x| Scalar::one() / (Scalar::one() + (-x).exp()))
    }

    fn activate_prime(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        let sigmoid = self.activate(input);
        sigmoid.clone() * (-sigmoid + Scalar::one())
    }
}

/// The Rectified Linear Unit activation function.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ReLU;

impl<Scalar> ActivationFn<Scalar> for ReLU
where
    Scalar: NdFloat,
{
    fn activate(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.map(|&value| value.max(Scalar::zero()))
    }

    fn activate_prime(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.map(|&value| {
            if value > Scalar::zero() {
                Scalar::one()
            } else {
                Scalar::zero()
            }
        })
    }
}

/// The leaky Rectified Linear Unit activation function. This is the same as the ReLU activation
/// function, except that the derivative is not zero when the input is negative.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LeakyReLU<Scalar>(pub Scalar);

impl Default for LeakyReLU<f32> {
    fn default() -> Self {
        Self(0.001)
    }
}

impl Default for LeakyReLU<f64> {
    fn default() -> Self {
        Self(0.001)
    }
}

impl<Scalar> ActivationFn<Scalar> for LeakyReLU<Scalar>
where
    Scalar: NdFloat,
{
    fn activate(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.map(|&value| {
            if value > Scalar::zero() {
                value
            } else {
                self.0 * value
            }
        })
    }

    fn activate_prime(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.map(|&value| {
            if value > Scalar::zero() {
                Scalar::one()
            } else {
                self.0
            }
        })
    }
}

/// The hyperbolic tangent activation function.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tanh;

impl<Scalar> ActivationFn<Scalar> for Tanh
where
    Scalar: NdFloat,
{
    fn activate(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.map(|&value| value.tanh())
    }

    fn activate_prime(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        let tanh = self.activate(input);
        let squared = &tanh * &tanh;
        -squared + Scalar::one()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    fn approx_eq(a: Array1<f64>, b: Array1<f64>) -> bool {
        dbg!(&a, &b);
        a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
    }

    #[test]
    fn sigmoid() {
        let sigmoid = Sigmoid;
        let input = array![0.0, 1.0, 2.0, 3.0];
        let output = sigmoid.activate(input);
        let expected = array![
            0.5,
            0.7310585786300049,
            0.8807970779778823,
            0.9525741268224334,
        ];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn sigmoid_prime() {
        let sigmoid = Sigmoid;
        let input = array![0.0, 1.0, 2.0, 3.0];
        let output = sigmoid.activate_prime(input);
        let expected = array![
            0.25,
            0.19661193324148185,
            0.10499358540350662,
            0.045176659730912995,
        ];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn relu() {
        let relu = ReLU;
        let input = array![0.0, 1.0, 2.0, 3.0];
        let output = relu.activate(input);
        let expected = array![0.0, 1.0, 2.0, 3.0];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn relu_prime() {
        let relu = ReLU;
        let input = array![0.0, 1.0, 2.0, 3.0];
        let output = relu.activate_prime(input);
        let expected = array![0.0, 1.0, 1.0, 1.0];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn leaky_relu() {
        let leaky_relu = LeakyReLU(0.001);
        let input = array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let output = leaky_relu.activate(input);
        let expected = array![-0.002, -0.001, 0.0, 1.0, 2.0, 3.0];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn leaky_relu_prime() {
        let leaky_relu = LeakyReLU(0.001);
        let input = array![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let output = leaky_relu.activate_prime(input);
        let expected = array![0.001, 0.001, 0.001, 1.0, 1.0, 1.0];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn tanh() {
        let tanh = Tanh;
        let input = array![0.0, 1.0, 2.0, 3.0];
        let output = tanh.activate(input);
        let expected = array![0.0f64.tanh(), 1.0f64.tanh(), 2.0f64.tanh(), 3.0f64.tanh()];
        assert!(approx_eq(output, expected));
    }

    #[test]
    fn tanh_prime() {
        let tanh = Tanh;
        let input = array![0.0, 1.0, 2.0, 3.0];
        let output = tanh.activate_prime(input);
        let expected = array![
            1.0 / 0.0f64.cosh().powf(2.0),
            1.0 / 1.0f64.cosh().powf(2.0),
            1.0 / 2.0f64.cosh().powf(2.0),
            1.0 / 3.0f64.cosh().powf(2.0),
        ];
        assert!(approx_eq(output, expected));
    }
}

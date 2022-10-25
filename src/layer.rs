use std::fmt::Debug;

use ndarray::{Array1, Array2, LinalgScalar};
use rand::{prelude::Distribution, RngCore};

use crate::LeakyReLU;

use super::{ActivationFn, ReLU, Sigmoid, Tanh};

/// A layer that supports forward propagation of values.
pub trait Forward<Scalar = f64> {
    /// Forward propagates the input.
    fn forward(&self, input: Array1<Scalar>) -> Array1<Scalar>;
}

impl<Scalar> Forward<Scalar> for Box<dyn Forward<Scalar>> {
    fn forward(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        self.as_ref().forward(input)
    }
}

/// A layer that can be updated by backpropagating gradients.
pub trait Layer<Scalar = f64>: Forward<Scalar> {
    /// Backpropagates the gradient. The input must be the value that was passed to
    /// [`Forward::forward`] to calculate the gradient.
    fn backward(
        &mut self,
        inputs: Array1<Scalar>,
        gradient: Array1<Scalar>,
        learning_rate: Scalar,
    ) -> Array1<Scalar>;
}

impl<Scalar> Forward<Scalar> for Box<dyn Layer<Scalar>> {
    fn forward(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        self.as_ref().forward(input)
    }
}

impl<Scalar> Layer<Scalar> for Box<dyn Layer<Scalar>> {
    fn backward(
        &mut self,
        inputs: Array1<Scalar>,
        gradient: Array1<Scalar>,
        learning_rate: Scalar,
    ) -> Array1<Scalar> {
        self.as_mut().backward(inputs, gradient, learning_rate)
    }
}

/// Extension trait for [`Layer`].
pub trait LayerExt: Layer + 'static {
    /// Boxes the layer dynamically.
    fn boxed(self) -> Box<dyn Layer>
    where
        Self: Sized,
    {
        Box::new(self)
    }
}

impl<L> LayerExt for L where L: Layer + Sized + 'static {}

/// A layer which applies an activation function to the given inputs.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ActivationLayer<A> {
    /// The activation function to apply.
    activation_fn: A,
}

impl<A> ActivationLayer<A> {
    /// Creates a new activation layer.
    pub const fn new(activation_fn: A) -> Self {
        Self { activation_fn }
    }

    /// Converts this activation layer into its inner activation function.
    pub fn into_inner(self) -> A {
        self.activation_fn
    }
}

impl ActivationLayer<Sigmoid> {
    /// Creates a new sigmoid activation layer.
    pub const fn sigmoid() -> Self {
        Self::new(Sigmoid)
    }
}

impl ActivationLayer<ReLU> {
    /// Creates a new ReLU activation layer.
    pub const fn relu() -> Self {
        Self::new(ReLU)
    }
}

impl<Scalar> ActivationLayer<LeakyReLU<Scalar>> {
    /// Creates a new leaky ReLU activation layer.
    pub const fn leaky_relu(alpha: Scalar) -> Self {
        Self::new(LeakyReLU(alpha))
    }
}

impl ActivationLayer<Tanh> {
    /// Creates a new tanh activation layer.
    pub const fn tanh() -> Self {
        Self::new(Tanh)
    }
}

impl<A, Scalar> Forward<Scalar> for ActivationLayer<A>
where
    A: ActivationFn<Scalar>,
{
    /// Applies `a(z)` to the input, where `a` is the activation function and `z` is the result of
    /// the previous layers (for example, the dot product of the input and the weights plus the
    /// biases).
    fn forward(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        self.activation_fn.activate(input)
    }
}

impl<A, Scalar> Layer<Scalar> for ActivationLayer<A>
where
    A: ActivationFn<Scalar>,
    Scalar: LinalgScalar,
{
    /// Propagates the gradient backwards through the activation function.
    fn backward(
        &mut self,
        inputs: Array1<Scalar>,
        gradient: Array1<Scalar>,
        _learning_rate: Scalar,
    ) -> Array1<Scalar> {
        // Calculate:
        // dC/dz = dC/da(z) * da(z)/dz

        // Given:
        // a(z) = self.forward(z)
        // z = input

        let da_dz = self.activation_fn.activate_prime(inputs);
        let dc_dz = gradient * da_dz;
        dc_dz
    }
}

/// A layer which outputs the dot product of the input and the weights, plus the biases.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DenseLayer<Scalar = f64> {
    /// A matrix of weights with the shape [input, output].
    pub weights: Array2<Scalar>,
    /// A vector of biases with the shape [outputs].
    pub biases: Array1<Scalar>,
}

impl<Scalar> DenseLayer<Scalar> {
    /// Creates a new dense layer with random weights and biases.
    pub fn new_random<R, D>(inputs: usize, outputs: usize, rng: &mut R, dist: D) -> Self
    where
        R: RngCore,
        D: Distribution<Scalar>,
    {
        let weights = Array2::from_shape_fn([inputs, outputs], |_| dist.sample(rng));
        let biases = Array1::from_shape_fn(outputs, |_| dist.sample(rng));
        Self { weights, biases }
    }
}

impl<Scalar> DenseLayer<Scalar>
where
    Scalar: LinalgScalar,
{
    /// Creates a new dense layer with zeros for weights and biases.
    pub fn new_zeros(inputs: usize, outputs: usize) -> Self {
        let weights = Array2::zeros([inputs, outputs]);
        let biases = Array1::zeros(outputs);
        Self { weights, biases }
    }
}

impl<Scalar> Forward<Scalar> for DenseLayer<Scalar>
where
    Scalar: LinalgScalar,
{
    /// Calculates the dot product of the input and the weights, plus the biases.
    fn forward(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        input.dot(&self.weights) + &self.biases
    }
}

impl Layer for DenseLayer {
    /// Calculates the gradient of the loss with respect to the inputs, and updates the weights and
    /// biases.
    fn backward(
        &mut self,
        inputs: Array1<f64>,
        gradient: Array1<f64>,
        learning_rate: f64,
    ) -> Array1<f64> {
        // Calculate:
        // dC/dw = dC/dz * dz/dw
        // dC/db = dC/dz * dz/db
        // dC/dinp = dC/dz * dz/dinp

        // Given:
        // dC/dz = gradient

        // Calculate gradient to backpropagate before weights are updated
        // z = inp * w + b
        // dz/dinp = w
        // dC/dinp = dC/dz * dz/dinp = dC/dz * w
        let dz_dinp = self.weights.t();
        let dc_dinp = gradient.dot(&dz_dinp);

        // Update biases
        // z = inp * w + b
        // dz/db = 1
        // dC/db = dC/dz * dz/db = gradient
        self.biases -= &(learning_rate * &gradient);

        // Update weights
        // z = inp * w + b
        // dz/dw = inp
        // dC/dw = dC/dz * dz/dw = gradient * dz_dw
        let delta_w = Array2::from_shape_fn(
            (self.weights.nrows(), self.weights.ncols()),
            |(inp_i, out_i)| learning_rate * gradient[out_i] * inputs[inp_i],
        );
        self.weights -= &delta_w;

        dc_dinp
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{array, Array, Dimension};

    use super::*;

    fn approx_eq<Ix>(actual: Array<f64, Ix>, expected: Array<f64, Ix>) -> bool
    where
        Ix: Dimension,
    {
        dbg!(&actual, &expected);
        actual
            .iter()
            .zip(expected.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    }

    #[test]
    fn test_relu_activation() {
        let mut layer = ActivationLayer::relu();
        let input = array![-1.0, 0.0, 1.0];
        let output = layer.forward(input.clone());
        assert!(approx_eq(output, array![0.0, 0.0, 1.0]));

        let gradient = array![1.0, 2.0, 3.0]; // dC/da
        let output_gradient = layer.backward(input, gradient, 1.0); // dC/dz = dC/da * da/dz = dC/da * a'(z)
        assert!(approx_eq(output_gradient, array![0.0, 0.0, 3.0]));
    }

    #[test]
    fn test_dense_forward() {
        let mut layer = DenseLayer::new_zeros(2, 3);
        layer.weights = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];
        layer.biases = array![1.0, 2.0, 3.0];

        let input = array![1.0, 2.0];
        let output = layer.forward(input);
        assert!(approx_eq(output, array![10.0, 14.0, 18.0]));
    }

    #[test]
    fn test_dense_backward() {
        let mut layer = DenseLayer::new_zeros(2, 3);
        layer.weights = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        layer.biases = array![1.0, 2.0, 3.0];

        let learning_rate = 0.1;
        let input = array![1.0, 2.0];
        let gradient = array![1.0, 2.0, 3.0]; // dC/dz
        let output_gradient = layer.backward(input, gradient, learning_rate); // dC/dinp = dC/dz * dz/dinp
        assert!(approx_eq(
            output_gradient,
            array![
                1.0 * 1.0 + 2.0 * 2.0 + 3.0 * 3.0,
                1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0
            ]
        ));
        assert!(approx_eq(
            layer.weights,
            array![
                [
                    1.0 - learning_rate * 1.0,
                    2.0 - learning_rate * 2.0,
                    3.0 - learning_rate * 3.0
                ],
                [
                    4.0 - learning_rate * 2.0,
                    5.0 - learning_rate * 4.0,
                    6.0 - learning_rate * 6.0
                ]
            ]
        ));
        assert!(approx_eq(
            layer.biases,
            array![
                1.0 - learning_rate * 1.0,
                2.0 - learning_rate * 2.0,
                3.0 - learning_rate * 3.0
            ]
        ));
    }
}

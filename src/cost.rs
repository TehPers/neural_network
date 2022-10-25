use ndarray::{Array1, ArrayView1, LinalgScalar};

/// A cost function.
pub trait CostFn<Scalar = f64> {
    /// Calculates the gradient from the prediction and real value.
    fn cost(&self, prediction: ArrayView1<Scalar>, real: ArrayView1<Scalar>) -> Scalar;

    /// Derivative of cost with respect to the prediction.
    fn cost_prime(
        &self,
        prediction: ArrayView1<Scalar>,
        real: ArrayView1<Scalar>,
    ) -> Array1<Scalar>;
}

impl<C, Scalar> CostFn<Scalar> for &C
where
    C: CostFn<Scalar>,
{
    fn cost(&self, prediction: ArrayView1<Scalar>, real: ArrayView1<Scalar>) -> Scalar {
        (*self).cost(prediction, real)
    }

    fn cost_prime(
        &self,
        prediction: ArrayView1<Scalar>,
        real: ArrayView1<Scalar>,
    ) -> Array1<Scalar> {
        (*self).cost_prime(prediction, real)
    }
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeanSquaredError;

impl<Scalar> CostFn<Scalar> for MeanSquaredError
where
    Scalar: LinalgScalar,
{
    fn cost(&self, prediction: ArrayView1<Scalar>, real: ArrayView1<Scalar>) -> Scalar {
        let diff = &prediction - &real;
        diff.dot(&diff)
    }

    fn cost_prime(
        &self,
        prediction: ArrayView1<Scalar>,
        real: ArrayView1<Scalar>,
    ) -> Array1<Scalar> {
        let diff = &prediction - &real;
        &diff + &diff
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn mse() {
        let mse = MeanSquaredError;
        let prediction = Array1::from(vec![0.0, 0.0, 0.0]);
        let real = Array1::from(vec![1.0, 2.0, 3.0]);
        assert_eq!(mse.cost(prediction.view(), real.view()), 14.0);
        assert_eq!(
            mse.cost_prime(prediction.view(), real.view()),
            array![-2.0, -4.0, -6.0]
        );
    }
}

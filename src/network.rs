use std::fmt::Debug;

use ndarray::Array1;

use crate::{
    Layer, MiniBatchTrainingOpts, TrainIter, TrainMiniBatchIter, TrainingExample, TrainingOpts,
};

use super::{CostFn, Forward};

/// A neural network.
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NeuralNetwork<L = Box<dyn Layer>> {
    /// The layers of the network.
    pub layers: Vec<L>,
}

impl<L> NeuralNetwork<L> {
    /// Creates a new model from the given layers.
    pub fn new(layers: Vec<L>) -> Self {
        Self { layers }
    }
}

impl<L> NeuralNetwork<L>
where
    L: Layer,
{
    /// Trains this model with a single training example. The returned iterator will perform a
    /// single epoch of training per iteration.
    pub fn train_single<'a, C>(
        &'a mut self,
        example: &'a TrainingExample,
        training_opts: &'a TrainingOpts<C>,
    ) -> TrainIter<'a, L, C>
    where
        C: CostFn,
    {
        TrainIter::new(self, example, training_opts)
    }

    /// Trains this model with a batch of training examples. The returned iterator will perform a
    /// single epoch of training per iteration.
    pub fn train_batch<'a, C>(
        &'a mut self,
        examples: &'a [TrainingExample],
        training_opts: &'a MiniBatchTrainingOpts<C>,
    ) -> TrainMiniBatchIter<'a, L, C>
    where
        C: CostFn,
    {
        TrainMiniBatchIter::new(self, examples, training_opts)
    }

    /// Trains this model with a batch of training examples in parallel. The returned iterator will
    /// perform a single epoch of training per iteration.
    #[cfg(feature = "parallel")]
    pub fn train_batch_par<'a, C>(
        &'a mut self,
        examples: &'a [TrainingExample],
        training_opts: &'a MiniBatchTrainingOpts<C>,
    ) -> crate::TrainParallelMiniBatchIter<'a, L, C>
    where
        C: CostFn,
    {
        crate::TrainParallelMiniBatchIter::new(self, examples, training_opts)
    }
}

impl<L, Scalar> Forward<Scalar> for NeuralNetwork<L>
where
    L: Forward<Scalar>,
{
    fn forward(&self, input: Array1<Scalar>) -> Array1<Scalar> {
        self.layers
            .iter()
            .fold(input, |input, layer| layer.forward(input))
    }
}

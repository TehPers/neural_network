use ndarray::Array1;
use rand::{seq::SliceRandom, thread_rng};

use crate::{CostFn, Layer, NeuralNetwork};

/// An example used for training a model.
#[derive(Clone, PartialEq, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingExample {
    /// The inputs to the model.
    pub input: Array1<f64>,
    /// The expected outputs of the model.
    pub output: Array1<f64>,
}

impl TrainingExample {
    /// Creates a new training example.
    pub fn new(input: Array1<f64>, output: Array1<f64>) -> Self {
        Self { input, output }
    }
}

/// Options for training a model.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrainingOpts<C> {
    /// The cost function for calculating the error.
    pub cost_fn: C,
    /// The learning rate.
    pub learning_rate: f64,
}

impl<C> Default for TrainingOpts<C>
where
    C: Default,
{
    fn default() -> Self {
        Self {
            cost_fn: Default::default(),
            learning_rate: 0.01,
        }
    }
}

/// A summary of the results from a single epoch.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EpochSummary {
    /// The total error for the epoch.
    pub loss: f64,
}

/// Trains a neural network on a single example. Each iteration of the iterator performs a single
/// training pass on the network.
pub struct TrainIter<'a, L, C> {
    network: &'a mut NeuralNetwork<L>,
    example: &'a TrainingExample,
    training_opts: &'a TrainingOpts<C>,
}

impl<'a, L, C> TrainIter<'a, L, C> {
    /// Creates a new iterator for training a network on a single example.
    pub fn new(
        network: &'a mut NeuralNetwork<L>,
        example: &'a TrainingExample,
        training_opts: &'a TrainingOpts<C>,
    ) -> Self {
        Self {
            network,
            example,
            training_opts,
        }
    }
}

impl<'a, L, C> Iterator for TrainIter<'a, L, C>
where
    L: Layer,
    C: CostFn,
{
    type Item = EpochSummary;

    fn next(&mut self) -> Option<Self::Item> {
        // Feedforward
        let mut inputs = Vec::with_capacity(self.network.layers.len());
        let mut input = self.example.input.clone();
        for layer in self.network.layers.iter() {
            inputs.push(input.clone());
            input = layer.forward(input);
        }

        // Calculate cost
        let loss = self
            .training_opts
            .cost_fn
            .cost(input.view(), self.example.output.view());
        let mut gradient = self
            .training_opts
            .cost_fn
            .cost_prime(input.view(), self.example.output.view());

        // Backpropagation
        for (layer, input) in self.network.layers.iter_mut().zip(inputs).rev() {
            gradient = layer.backward(input, gradient, self.training_opts.learning_rate);
        }

        let summary = EpochSummary { loss };
        Some(summary)
    }
}

/// Options for training a model using mini-batches.
#[derive(Clone, Debug)]
pub struct MiniBatchTrainingOpts<C> {
    /// The cost function for calculating the error.
    pub cost_fn: C,
    /// The learning rate.
    pub learning_rate: f64,
    /// The size of each mini-batch.
    pub batch_size: usize,
}

/// Trains a neural network on batches of examples. Each iteration of the iterator performs a
/// single training pass on the network.
#[derive(Debug)]
pub struct TrainMiniBatchIter<'a, L, C> {
    network: &'a mut NeuralNetwork<L>,
    examples: &'a [TrainingExample],
    training_opts: &'a MiniBatchTrainingOpts<C>,
    example_indices: Vec<usize>,
}

impl<'a, L, C> TrainMiniBatchIter<'a, L, C> {
    /// Creates a new iterator for training a network on batches of examples.
    pub fn new(
        network: &'a mut NeuralNetwork<L>,
        examples: &'a [TrainingExample],
        training_opts: &'a MiniBatchTrainingOpts<C>,
    ) -> Self {
        let example_indices = (0..examples.len()).collect();
        Self {
            network,
            examples,
            training_opts,
            example_indices,
        }
    }
}

impl<'a, L, C> Iterator for TrainMiniBatchIter<'a, L, C>
where
    L: Layer,
    C: CostFn,
{
    type Item = EpochSummary;

    fn next(&mut self) -> Option<Self::Item> {
        // Shuffle examples
        let mut rng = thread_rng();
        self.example_indices.shuffle(&mut rng);

        // Train on each batch
        let mut total_loss = 0.0;
        for batch in self.example_indices.chunks(self.training_opts.batch_size) {
            let mut gradients = Vec::with_capacity(self.training_opts.batch_size);
            let mut batch_inputs = Vec::with_capacity(self.training_opts.batch_size);
            for &example_index in batch {
                let example = &self.examples[example_index];

                // Feedforward
                let mut inputs = Vec::with_capacity(self.network.layers.len());
                let output =
                    self.network
                        .layers
                        .iter()
                        .fold(example.input.clone(), |input, layer| {
                            inputs.push(input.clone());
                            layer.forward(input)
                        });
                batch_inputs.push(inputs);

                // Calculate cost
                let loss = self
                    .training_opts
                    .cost_fn
                    .cost(output.view(), example.output.view());
                let gradient = self
                    .training_opts
                    .cost_fn
                    .cost_prime(output.view(), example.output.view());

                total_loss += loss;
                gradients.push(gradient);
            }

            // Backpropagation
            for (gradient, inputs) in gradients.into_iter().zip(batch_inputs) {
                self.network.layers.iter_mut().zip(inputs).rev().fold(
                    gradient,
                    |gradient, (layer, input)| {
                        let learning_rate = self.training_opts.learning_rate / batch.len() as f64;
                        layer.backward(input, gradient, learning_rate)
                    },
                );
            }
        }

        let summary = EpochSummary { loss: total_loss };
        Some(summary)
    }
}

/// Trains a neural network on batches of examples, using thread parallelism where possible. Each
/// iteration of the iterator performs a single training pass on the network.
#[derive(Debug)]
#[cfg(feature = "parallel")]
pub struct TrainParallelMiniBatchIter<'a, L, C> {
    network: &'a mut NeuralNetwork<L>,
    examples: &'a [TrainingExample],
    training_opts: &'a MiniBatchTrainingOpts<C>,
    example_indices: Vec<usize>,
}

#[cfg(feature = "parallel")]
impl<'a, L, C> TrainParallelMiniBatchIter<'a, L, C> {
    /// Creates a new iterator for training a network on batches of examples in parallel.
    pub fn new(
        network: &'a mut NeuralNetwork<L>,
        examples: &'a [TrainingExample],
        training_opts: &'a MiniBatchTrainingOpts<C>,
    ) -> Self {
        let example_indices = (0..examples.len()).collect();
        Self {
            network,
            examples,
            training_opts,
            example_indices,
        }
    }
}

#[cfg(feature = "parallel")]
impl<'a, L, C> Iterator for TrainParallelMiniBatchIter<'a, L, C>
where
    L: Layer + Sync,
    C: CostFn + Sync,
{
    type Item = EpochSummary;

    fn next(&mut self) -> Option<Self::Item> {
        use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

        // Shuffle examples
        let mut rng = thread_rng();
        self.example_indices.shuffle(&mut rng);

        // Train on each batch
        let mut total_loss = 0.0;
        for batch in self.example_indices.chunks(self.training_opts.batch_size) {
            let (batch_inputs, batch_loss, batch_gradients) = batch
                .par_iter()
                .map(|&i| {
                    let example = &self.examples[i];

                    // Feedforward
                    let mut inputs = Vec::with_capacity(self.network.layers.len());
                    let output =
                        self.network
                            .layers
                            .iter()
                            .fold(example.input.clone(), |input, layer| {
                                inputs.push(input.clone());
                                layer.forward(input)
                            });

                    // Calculate cost
                    let loss = self
                        .training_opts
                        .cost_fn
                        .cost(output.view(), example.output.view());
                    let gradient = self
                        .training_opts
                        .cost_fn
                        .cost_prime(output.view(), example.output.view());

                    (inputs, loss, gradient)
                })
                .fold(
                    || {
                        (
                            Vec::with_capacity(batch.len()),
                            0.0,
                            Vec::with_capacity(batch.len()),
                        )
                    },
                    |(mut batch_inputs, mut batch_loss, mut batch_gradients),
                     (single_inputs, single_loss, single_gradient)| {
                        batch_inputs.push(single_inputs);
                        batch_loss += single_loss;
                        batch_gradients.push(single_gradient);
                        (batch_inputs, batch_loss, batch_gradients)
                    },
                )
                .reduce(
                    || {
                        (
                            Vec::with_capacity(batch.len()),
                            0.0,
                            Vec::with_capacity(batch.len()),
                        )
                    },
                    |(mut batch_inputs1, mut batch_loss1, mut batch_gradients1),
                     (batch_inputs2, batch_loss2, batch_gradients2)| {
                        batch_inputs1.extend(batch_inputs2);
                        batch_loss1 += batch_loss2;
                        batch_gradients1.extend(batch_gradients2);
                        (batch_inputs1, batch_loss1, batch_gradients1)
                    },
                );

            // Backpropagation
            for (inputs, gradient) in batch_inputs.into_iter().zip(batch_gradients) {
                self.network.layers.iter_mut().zip(inputs).rev().fold(
                    gradient,
                    |gradient, (layer, input)| {
                        layer.backward(
                            input,
                            gradient,
                            self.training_opts.learning_rate / batch.len() as f64,
                        )
                    },
                );
            }

            total_loss += batch_loss;
        }

        let summary = EpochSummary { loss: total_loss };
        Some(summary)
    }
}

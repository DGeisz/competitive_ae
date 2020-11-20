use mccm::{MnistNetwork, MnistNeuron, MNIST_SIDE};
use std::cell::RefCell;
use std::rc::Rc;

pub struct NeuronicInput {
    measure: RefCell<f32>,
    total_weighted_prediction: RefCell<f32>,
    weight_holder: Rc<WeightHolder>,
    current_reconstruction_error: RefCell<f32>,
}

impl NeuronicInput {
    pub fn new(weight_holder: Rc<WeightHolder>) -> NeuronicInput {
        NeuronicInput {
            measure: RefCell::new(0.0),
            total_weighted_prediction: RefCell::new(0.0),
            weight_holder,
            current_reconstruction_error: RefCell::new(0.0),
        }
    }

    pub fn get_measure(&self) -> f32 {
        *self.measure.borrow()
    }

    pub fn load_input_measure(&self, measure: f32) {
        *self.measure.borrow_mut() = measure;
    }

    pub fn incr_total_weighted_prediction(&self, weighted_prediction: f32) {
        *self.total_weighted_prediction.borrow_mut() += weighted_prediction;
    }

    pub fn get_reconstruction(&self) -> f32 {
        *self.total_weighted_prediction.borrow() / self.weight_holder.get_total_weight()
    }

    /// Caches reconstruction error to speed up
    /// future error lookups
    pub fn cache_reconstruction_error(&self) {
        let reconstruction =
            *self.total_weighted_prediction.borrow() / self.weight_holder.get_total_weight();
        *self.current_reconstruction_error.borrow_mut() = reconstruction - *self.measure.borrow();
    }

    /// This is the signed error.  Only the magnitude is important
    /// for the size of the error, but the sign is necessary for
    /// learning weights
    pub fn get_reconstruction_error(&self) -> f32 {
        *self.current_reconstruction_error.borrow()
    }
}

/// If you distributed this network, then each input would hold
/// the total input weights, but it's redundant in the present case
pub struct WeightHolder {
    total_weights: RefCell<f32>,
}

impl WeightHolder {
    pub fn new() -> WeightHolder {
        WeightHolder {
            total_weights: RefCell::new(0.0),
        }
    }

    pub fn clear(&self) {
        *self.total_weights.borrow_mut() = 0.0;
    }

    pub fn incr_weight(&self, weight: f32) {
        *self.total_weights.borrow_mut() += weight;
    }

    pub fn get_total_weight(&self) -> f32 {
        *self.total_weights.borrow()
    }
}

pub struct CompAENeuron {
    name: String,
    learning_constant: f32,
    inputs: Vec<Rc<NeuronicInput>>,
    weights: Vec<RefCell<f32>>,
    weight_holder: Rc<WeightHolder>,
    current_em: RefCell<f32>,
}

impl CompAENeuron {
    pub fn run_prediction_phase(&self) {
        let mut current_em = self.current_em.borrow_mut();

        *current_em = self.compute_em();
        self.weight_holder.incr_weight(*current_em);

        for (input, weight) in self.inputs.iter().zip(self.weights.iter()) {
            input.incr_total_weighted_prediction(*current_em * *weight.borrow());
        }
    }

    pub fn run_learning_phase(&self) {
        let adjustment_size = (*self.current_em.borrow() / self.weight_holder.get_total_weight())
            * self.learning_constant;

        for (input, weight) in self.inputs.iter().zip(self.weights.iter()) {
            *weight.borrow_mut() += -1.0 * input.get_reconstruction_error() * adjustment_size;
        }
    }
}

impl MnistNeuron for CompAENeuron {
    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn compute_em(&self) -> f32 {
        let mut total_weight = 0.0;
        let mut total_weighted_em = 0.0;

        for (input, weight_ref) in self.inputs.iter().zip(self.weights.iter()) {
            let weight = *weight_ref.borrow_mut();

            total_weight += weight;
            total_weighted_em += weight * input.get_measure();
        }

        total_weighted_em / total_weight
    }
}

pub struct CompAENetwork {
    neurons: Vec<Rc<CompAENeuron>>,
    inputs: Vec<Rc<NeuronicInput>>,
    weight_holder: Rc<WeightHolder>,
    learning_constant: f32,
}

impl MnistNetwork for CompAENetwork {
    fn get_neurons(&self) -> Vec<Rc<dyn MnistNeuron>> {
        self.neurons
            .iter()
            .map(|neuron| Rc::clone(neuron) as Rc<dyn MnistNeuron>)
            .collect()
    }

    /// Clears the weight holder if it loads (0, 0) because
    /// the weight holder needs to be cleared once during
    /// the loading phase
    fn load_val(&self, x: usize, y: usize, val: f32) {
        let input_index = (MNIST_SIDE * y) + x;
        let input = self.inputs.get(input_index).unwrap();
        input.load_input_measure(val);

        if (x, y) == (0, 0) {
            self.weight_holder.clear();
        }
    }

    fn perform_adjustment(&mut self) {
        // Reconstruction phase
        for neuron in self.neurons.iter_mut() {
            neuron.run_prediction_phase();
        }

        // Cache the reconstruction error for speedy lookup
        for input in &self.inputs {
            input.cache_reconstruction_error();
        }

        // Run learning phase
        for neuron in self.neurons.iter_mut() {
            neuron.run_learning_phase();
        }
    }
}

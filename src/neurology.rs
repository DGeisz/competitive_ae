use mccm::{MnistNetwork, MnistNeuron, MNIST_SIDE};
use std::cell::Cell;
use std::rc::Rc;
use std::path::Path;
use std::fs::File;
use std::io::Write;
use rand::Rng;

pub struct NeuronicInput {
    measure: Cell<f32>,
    total_weighted_prediction: Cell<f32>,
    weight_holder: Rc<WeightHolder>,
    current_reconstruction_error: Cell<f32>,
}

impl NeuronicInput {
    pub fn new(weight_holder: Rc<WeightHolder>) -> NeuronicInput {
        NeuronicInput {
            measure: Cell::new(0.0),
            total_weighted_prediction: Cell::new(0.0),
            weight_holder,
            current_reconstruction_error: Cell::new(0.0),
        }
    }

    pub fn get_measure(&self) -> f32 {
        self.measure.get()
    }

    pub fn load_input_measure(&self, measure: f32) {
        self.measure.replace(measure);
    }

    pub fn incr_total_weighted_prediction(&self, weighted_prediction: f32) {
        self.total_weighted_prediction.replace(self.total_weighted_prediction.get() + weighted_prediction);
    }

    pub fn clear_total_weighted_prediction(&self) {
        self.total_weighted_prediction.replace(0.0);
    }

    pub fn get_reconstruction(&self) -> f32 {
        self.total_weighted_prediction.get() / self.weight_holder.get_total_weight()
    }

    /// Caches reconstruction error to speed up
    /// future error lookups
    pub fn cache_reconstruction_error(&self) {
        let reconstruction =
            self.total_weighted_prediction.get() / self.weight_holder.get_total_weight();
        self.current_reconstruction_error.replace(reconstruction - self.measure.get());
    }

    /// This is the signed error.  Only the magnitude is important
    /// for the size of the error, but the sign is necessary for
    /// learning weights
    pub fn get_reconstruction_error(&self) -> f32 {
        self.current_reconstruction_error.get()
    }
}

/// If you distributed this network, then each input would hold
/// the total input weights, but it's redundant in the present case
pub struct WeightHolder {
    total_weights: Cell<f32>,
}

impl WeightHolder {
    pub fn new() -> WeightHolder {
        WeightHolder {
            total_weights: Cell::new(0.0),
        }
    }

    pub fn clear(&self) {
        self.total_weights.replace(0.0);
    }

    pub fn incr_weight(&self, weight: f32) {
        self.total_weights.replace(self.total_weights.get() + weight);
    }

    pub fn get_total_weight(&self) -> f32 {
        1.0
    }
}

pub struct CompAENeuron {
    name: String,
    learning_constant: f32,
    inputs: Vec<Rc<NeuronicInput>>,
    weights: Vec<Cell<f32>>,
    weight_holder: Rc<WeightHolder>,
    current_em: Cell<f32>,
}

impl CompAENeuron {
    pub fn new(
        name: String,
        learning_constant: f32,
        gen_weight: fn() -> f32,
        inputs: Vec<Rc<NeuronicInput>>,
        weight_holder: Rc<WeightHolder>,
    ) -> CompAENeuron {
        let weights = (0..inputs.len())
            .map(|_| Cell::new(gen_weight()))
            .collect::<Vec<Cell<f32>>>();

        CompAENeuron {
            name,
            learning_constant,
            inputs,
            weights,
            weight_holder,
            current_em: Cell::new(0.0),
        }
    }

    pub fn run_prediction_phase(&self) {
        let em = self.compute_em();
        self.current_em.replace(em);
        self.weight_holder.incr_weight(em);

        for (input, weight) in self.inputs.iter().zip(self.weights.iter()) {
            input.incr_total_weighted_prediction(em * weight.get());
        }
    }

    pub fn run_learning_phase(&self) {
        let adjustment_size = (self.current_em.get() / self.weight_holder.get_total_weight())
            * self.learning_constant;

        for (input, weight) in self.inputs.iter().zip(self.weights.iter()) {
            weight.replace(weight.get() + (-1.0 * input.get_reconstruction_error() * adjustment_size));

            if weight.get() < 0.0 {
                weight.replace(0.0);
            }
        }
    }

    pub fn to_serializable(&self) -> Vec<Vec<f32>> {
        let mut val_matrix = Vec::new();

        for j in 0..MNIST_SIDE {
            let mut val_row = Vec::new();

            for i in 0..MNIST_SIDE {
                val_row.push(self.weights.get((j * MNIST_SIDE) + i).unwrap().get());
            }

            val_matrix.push(val_row);
        }

        val_matrix
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
            let weight = weight_ref.get();

            total_weight += weight;
            total_weighted_em += weight * input.get_measure();
        }

        total_weighted_em / total_weight.sqrt()
        // total_weighted_em
    }
}

pub struct CompAENetwork {
    neurons: Vec<Rc<CompAENeuron>>,
    inputs: Vec<Rc<NeuronicInput>>,
    weight_holder: Rc<WeightHolder>,
}

impl CompAENetwork {
    pub fn new(learning_constant: f32, num_neurons: usize, num_inputs: usize, gen_synapse_weight: fn() -> f32) -> CompAENetwork {
        let weight_holder = Rc::new(WeightHolder::new());

        // Initialize inputs
        let inputs = (0..num_inputs)
            .map(|_| Rc::new(NeuronicInput::new(Rc::clone(&weight_holder))))
            .collect::<Vec<Rc<NeuronicInput>>>();

        let neurons = (0..num_neurons)
            .map(|i| {
                Rc::new(CompAENeuron::new(
                    i.to_string(),
                    learning_constant,
                    gen_synapse_weight,
                    inputs.iter().map(|input| Rc::clone(input)).collect(),
                    Rc::clone(&weight_holder),
                ))
            })
            .collect::<Vec<Rc<CompAENeuron>>>();

        CompAENetwork {
            neurons,
            inputs,
            weight_holder
        }
    }

    pub fn serialize(&self) {
        let py_data: Vec<Vec<Vec<f32>>> = self.neurons.iter().map(|n| n.to_serializable()).collect();

        let pickle = serde_pickle::to_vec(&py_data, true).unwrap();

        let path = Path::new("./output_data/data.pickle");
        let mut file = File::create(path).unwrap();

        file.write_all(&pickle).unwrap();
    }
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
        input.clear_total_weighted_prediction();

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

use mccm::{MnistNetwork, MNIST_AREA};
use mnist::{Mnist, MnistBuilder};
use rand::Rng;
use crate::neurology::CompAENetwork;

mod neurology;

const NUM_NEURONS: usize = 10;
const LEARNING_CONST: f32 = 0.001;
const EPOCHS: usize = 10;

const MIN_INIT_WEIGHT: f32 = 0.0;
const MAX_INIT_WEIGHT: f32 = 1.0 / NUM_NEURONS as f32;

const TRAINING_SET_LENGTH: u32 = 10000;
const TEST_SET_LENGTH: u32 = 10000;

const LOGGER_ON: bool = true;

fn generate_weight() -> f32 {
    rand::thread_rng().gen_range(MIN_INIT_WEIGHT, MAX_INIT_WEIGHT)
}

fn main() {
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(TRAINING_SET_LENGTH)
        .validation_set_length(0)
        .test_set_length(TEST_SET_LENGTH)
        .finalize();

    let train_img: Vec<f32> = trn_img.iter().map(|val| *val as f32 / 255.0).collect();
    let test_img: Vec<f32> = tst_img.iter().map(|val| *val as f32 / 255.0).collect();

    let mut network = CompAENetwork::new(LEARNING_CONST, NUM_NEURONS, MNIST_AREA, generate_weight);

    let accuracy = network.take_metric(train_img, trn_lbl, EPOCHS, test_img, tst_lbl, LOGGER_ON);

    println!("Model accuracy: {}", accuracy);

    network.serialize();
}

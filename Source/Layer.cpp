#include "../Headers/Layer.hpp"

// Debug
#include <exception>
#include <iostream>
#include <stdexcept>

// Constructors

// Create a layer
Layer::Layer(layer_s layerSize, layer_s nextLayerSize, LayerType type) : type(type) {
	// Initialize the weights to random values
	InitializeWeights(layerSize, nextLayerSize);

	// Inhitialize the biases to random values
	InitializeBiases(nextLayerSize);

	// Initialize the neurons to 0
	InitializeNeurons(layerSize);
}

// Overload for the output layer, since it has no weights
Layer::Layer(layer_s layerSize, LayerType type) : type(type) {
	// Initialize the neurons to 0
	InitializeNeurons(layerSize);
}

// Function to generate a random weight/bias. Only used in the InitializeWeights function
float generateWeight() {
	return fmod(random()/(float)INITIAL_WEIGHT_ACCURACY, INITIAL_WEIGHT_RANGE*2) -INITIAL_WEIGHT_RANGE;
}

// Private Methods

// Initialize the weights to a random float between -INITIAL_WEIGHT_RANGE and INITIAL_WEIGHT_RANGE
void Layer::InitializeWeights(layer_s layerSize, layer_s nextLayerSize) {
	// Loop through each row of the matrix
	for(layer_s row = 0; row < nextLayerSize; row++) {

		// Populate the row with a "vector"
		Vector vector;
		// Populate the "vector" with random weights
		for (layer_s column = 0; column < layerSize; column++) {
			vector.push_back(generateWeight());
		}

		// Set the column of the matrix to the new "vector"
		this->weights.push_back(vector);
	}
}

// Initialize neurons with a value of 0 (OFF)
void Layer::InitializeNeurons(layer_s layerSize) {
	for (int i = 0; i < layerSize; i++) {
		this->neurons.push_back(0);
	}
}

// Initialize biases to random values
void Layer::InitializeBiases(layer_s biasCount) {
	for (int i = 0; i < biasCount; i++) {
		this->biases.push_back(generateWeight());
	}
}

// Getters
Matrix Layer::getWeights() {
	return this->weights;
}

Vector Layer::getNeurons() {
	return this->neurons;
}

Vector Layer::getBias() {
	return this->biases;
}

LayerType Layer::getType() {
	return this->type;
}

// Setters

// Set the neurons
void Layer::setNeurons(Vector newNeurons) {
	this->neurons = newNeurons;
}

// Methods

// Will throw std::domain_error if attempting to find layer after output layer
void Layer::setNextNeurons(Layer& nextLayer) {
	if (this->type == Output) {
		throw std::domain_error("Attempting to find layer after output layer");
	}

	// Size of neuron vector, will be the column count of matrix

	int nextLayerSize = this->weights.size();
	int layerSize = this->neurons.size();

	Vector nextLayerNeurons;

	// Loop nextLayerSize times, to populate nextLayer
	// Do matrix multiplication, basically
	for(int i = 0; i < nextLayerSize; i++) {
		float sum = 0;
		for (int l = 0; l < layerSize; l++) {
			sum += this->neurons[l] * this->weights[i][l];
		}
		sum += this->biases[i];

		// Sigmoid squishification function, and set the next node equal
		nextLayerNeurons.push_back(ReLU(sum));
	}

	nextLayer.setNeurons(nextLayerNeurons);
}
#include "../Headers/NeuralNetwork.hpp"
#include <cmath>

NeuralNetwork::NeuralNetwork(int inputCount, int floatingCount, int outputCount) :
	Network({inputCount, floatingCount, outputCount}) {}

// Backpropagation method
Vector NeuralNetwork::train(Vector input, Vector expectedOutput) {
	// // concat the input values to the end of the floating values matrix.
	// // This matrix just gives the output values, so since we want to
	// // relate the input directly to the output in addition to through neuron paths...
	// Vector floatingAndInput = input;
	// for (float x : this->floatingValues)
	// 	floatingAndInput.push_back(x);

	// // Calculate the floating layer
	// this->floatingValues = calculateLayer(floatingAndInput, this->inputWeights, this->floatingBiases);

	// Vector output = calculateLayer(floatingAndInput, this->floatingWeights, this->outputBiases);



	// // Adjust the connections to the output neurons
	// Vector cost = Network::calculateCost(output, expectedOutput);
	// // Step 1: Adjust Biases for Output Neurons
	// adjustBiases(&this->outputBiases, &cost);

	// //Step 2: Adjust Weights before Output Neurons
	// adjustWeights(&this->floatingWeights, &cost, &floatingAndInput);
	
	// // Step 3: Adjust Neurons
	// Vector floatingExpected = adjustNeurons(&this->floatingWeights, &cost, &this->floatingValues);
	


	// // // Adjust the connections to the floating neurons
	// cost = Network::calculateCost(this->floatingValues, floatingExpected);
	// // // Step 1: Adjust Biases for Floating Neurons
	// adjustBiases(&this->floatingBiases, &cost);

	// // // Step 2: Adjust Weights before Floating Neurons
	// adjustWeights(&this->inputWeights, &cost, &floatingAndInput);


	// // Return the output in case the user wants to use it
	// return output;
}

void NeuralNetwork::adjustBiases(Vector* biases, Vector* cost) {
	for (int i=0;i<biases->size();i++) {
		(*biases)[i] += (*cost)[i] / BIAS_ADJUST_DIVISOR;
	}
}

void NeuralNetwork::adjustWeights(Matrix* weights, Vector* cost, Vector* values) {
	for (int r=0;r<weights->size();r++) { // For each output neuron
		
		for (int n=0;n<values->size();n++) { // For each input neuron / connection to selected output neuron
			// The wcost is the difference between the cost and the value.
			float wcost = (*cost)[r] * (*values)[n];
			(*weights)[r][n] += wcost / WEIGHT_ADJUST_DIVISOR;
		}
	}
}

Vector NeuralNetwork::adjustNeurons(Matrix* weights, Vector* cost, Vector* values) {
	// The list of adjustments to apply to 
	Vector neuronAdjustments(values->size());

	for (int r=0;r<weights->size();r++) { // For each output neuron
		
		for (int n=0;n<values->size();n++) { // For each input neuron / connection to selected output neuron
			// The wcost is the difference between the cost and the value.

			float wcost = (*cost)[r] * (*values)[n];
			neuronAdjustments[n] += wcost / NEURON_ADJUST_DIVISOR;
		}
	}

	return neuronAdjustments;
}
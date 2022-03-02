#include "../Headers/Network.hpp"

// debug
#include <iostream>
#include <math.h>
#include <stdexcept>

// Create a network
// Layer sizes are passed by using array syntax as parameter
// Input, Hidden layers, Output
// Ex. Layer myLayer(2, {3, 3}, 2);
Network::Network(layer_s input, SizeList layerSizes, layer_s output) {
	// Get the layer size from input

	// Attempt to reserve size for all our layers
	try {
		this->layers.reserve(layerSizes.size() + 2);
	}
	catch (const std::length_error& le) {
		std::cerr << "Length error: " << le.what() << std::endl;
		std::cerr << "Too many hidden layers" << std::endl;
		exit(1);
	}

	// Seed the pseudorandom number generator to the current time in seconds
	srand(time(NULL));


	// Add all the sizes to an array so we know the size of the next layer,
	// so we can figure out the size of the weight matrix
	layer_s* sizeList = (layer_s*)malloc(sizeof(layer_s) * (layerSizes.size() + 2));

	sizeList[0] = input;
	int counter = 0;
	for (layer_s i : layerSizes) {
		sizeList[++counter] = i;
	}
	sizeList[++counter] = output;

	// Reset the counter
	counter = 0;

	// Add the input layer
	this->layers.push_back(Layer(input, sizeList[++counter], Input));
	// Add all the hidden layers
	for (layer_s i : layerSizes) {
		this->layers.push_back(Layer(i, sizeList[++counter], Hidden));
	}
	// Add the output layer
	this->layers.push_back(Layer(output, Output));

	free(sizeList);
}

// Methods

// Give the network some input, receive output
Vector Network::askNetwork(Vector inputNodes) {
	// Set the neurons for the input layer
	this->layers[0].setNeurons(inputNodes);

	// Propagate through input layer and all hidden layers
	for (int i = 0; i < this->layers.size() - 1; i++) {
		this->layers[i].setNextNeurons(this->layers[i+1]);
	}

	return this->layers.back().getNeurons();
}

// Train the network (Begin backpropagation)
Vector Network::train(Vector input, Vector expectedOutput) {
	Vector actualOutput = this->askNetwork(input);

	// Begin the recursive backpropagation algorithm
	this->backPropagate(expectedOutput, actualOutput, this->layers.size()-1);

	return actualOutput;
}

// A recursive implementation of the backpropagation algorithm
void backPropagate(Vector expectedOutput, Vector actualOutput, int layer) {

}

// Perform a Mean Square Error calculation
float Network::calculateNetworkCost(Vector expectedOutput, Vector actualOutput) {
	if (expectedOutput.size() != actualOutput.size())
		throw std::invalid_argument("Expected output and actual output do not match in size");
	if (expectedOutput.size() < 1)
		throw std::invalid_argument("Argument vectors must be larger than 0");

	// Mean Square error caluclation
	float sum = 0;
	for (int i = 0; i < expectedOutput.size(); i++) {
		sum += localCost(actualOutput[i], expectedOutput[i]);
	}
	sum *= 1.f/expectedOutput.size();

	// Return the cost
	return sum;
}

// Print the neurons given
void Network::PrintNeurons(Vector neurons) {
	std::cout << "[ ";
	for (float i : neurons) {
		std::cout << i << " ";
	}
	std::cout << "]" << std::endl;
}
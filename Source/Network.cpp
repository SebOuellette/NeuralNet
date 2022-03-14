#include "../Headers/Network.hpp"

// debug
#include <iostream>
//#define DEBUG_MODE

// Higher number protects against bias overcorrection
// Log of the best divisor for different hidden tree layouts:
// 2, 2, 2 - 4 or 15 or 6
// 5, 7, 5 - 2
// 8, 4, 8 - 4
// 3, 3, 3 - 4 or 9
// 4, 4, 4 - 10 or maybe 12
#define SHIFT_DIVISOR 2

// Unsure of the concequences of this multiplier.... 300 seems to make training work faster for most cases
#define BACKPROP_COST_MULTIPLIER 300

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
#ifdef DEBUG_MODE
	std::cout << "DEBUG MODE ENABLED" << std::endl;
	srand(2);
#else
	srand(time(NULL));
#endif
	


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

	//this->display();

	// Begin the recursive backpropagation algorithm
	this->backPropagate(expectedOutput, actualOutput, this->layers.size()-1);

	return actualOutput;
}

// Get the weights connecting the previous layer with the current neuron
Vector Network::getPreviousWeights(index currentLayer, index neuron) {
	if (currentLayer < 1)
		throw std::invalid_argument("Current layer has now previous layers");
	if (neuron < 0)
		throw std::invalid_argument("Neuron index is less than 0");

	// neuronth row is the weights connecting the previous layer to that neuron
	return this->layers[currentLayer-1].getWeights()[neuron];
}

// Get the weights connecting the previous layer with the current neuron
float Network::getPreviousBias(index currentLayer, index neuron) {
	if (currentLayer < 1)
		throw std::invalid_argument("Current layer has now previous layers");
	if (neuron < 0)
		throw std::invalid_argument("Neuron index is less than 0");

	// neuronth row is the bias to find that neuron
	return this->layers[currentLayer-1].getBiases()[neuron];
}

// A recursive implementation of the backpropagation algorithm
void Network::backPropagate(Vector expectedOutput, Vector actualOutput, index layer) {
	// Exit condition
	if (this->layers[layer].getType() == Input)
		return;

	int neuronCount = this->layers[layer].getNeurons().size();
	Vector previousNeurons = this->layers[layer-1].getNeurons();

	// The desired changes for the previous layer of neurons
	Vector desiredNeuronChanges;
	// Fill the vector with 0s.
	for(int i = 0; i < previousNeurons.size(); i++)
		desiredNeuronChanges.push_back(0);

	// Loop through all neurons
	for (index n = 0; n < neuronCount; n++) {
		float cost = localCost(expectedOutput[n], actualOutput[n]);

		// Step 1: Proportionally to the Cost, Change the Bias
		// This is working
		this->layers[layer-1].moveBias(n, cost / SHIFT_DIVISOR);

		// Step 2: Change the Weights
		// Loop through all the prevoius neurons
		for (int i = 0; i < previousNeurons.size(); i++) {
			// Find the cost between expected and value of previous neuron
			float previousNeuronCost = previousNeurons[i] * cost;

			// Adjust weight based on that cost
			this->layers[layer-1].moveWeight(n, i, previousNeuronCost / SHIFT_DIVISOR);
			

			// For step 3, save all the desired changes for the next neuron
			desiredNeuronChanges[i] += abs((this->layers[layer-1].getWeights()[n][i])) * cost * BACKPROP_COST_MULTIPLIER;
		}
	}

	// Step 3: Change the Neurons
	// Add the previous layer's neurons to the desired neuron changes
	if (layer != 1) {
		for (int i = 0; i < previousNeurons.size(); i++) {
			desiredNeuronChanges[i] = ReLU(weightBiasClamp(desiredNeuronChanges[i]) + previousNeurons[i]);
			
		}
	}

	// Recursively call this function again, as the final part of Step 3
	this->backPropagate(desiredNeuronChanges, previousNeurons, layer - 1);
}

// Convert the entire neural network into text
void Network::display() {
	// Display a simplistic view of the model
	for (int i=0;i<this->layers.size();i++) {
		std::cout << this->layers[i].getNeurons().size() << ((i != this->layers.size()-1) ? " : " : "");
	}
	std::cout << std::endl;

	// Display all the values associated to each neuron, weight, and bias
	for (int i = 0 ; i < this->layers.size()-1; i++) {
		// display the neurons
		if (i == 0)
			std::cout << "Input: ";
		else
		 	std::cout << i << ": ";
		Network::PrintNeurons(this->layers[i].getNeurons());

		std::cout << std::endl;

		// Display the weights
		std::cout << "Weights: " << std::endl;
		Network::PrintMatrix(this->layers[i].getWeights());

		// Display the biases
		std::cout << "Biases: " << std::endl;
		Network::PrintNeurons(this->layers[i].getBiases());

		std::cout << std::endl;
	}

	std::cout << "Output: ";
	Network::PrintNeurons(this->layers.back().getNeurons());
}

// Perform a Mean Square Error calculation
float Network::calculateNetworkCost(Vector expectedOutput, Vector actualOutput) {
	if (expectedOutput.size() != actualOutput.size())
		throw std::invalid_argument("Expected output and actual output do not match in size");
	if (expectedOutput.size() < 1)
		throw std::invalid_argument("Argument vectors must be larger than 0");

	// Mean Square error caluclation - Customized
	// It's more like Mean Cubed Error
	float sum = 0;
	for (int i = 0; i < expectedOutput.size(); i++) {
		sum += fabs(localCost(actualOutput[i], expectedOutput[i]));
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

// Print a matrix
void Network::PrintMatrix(Matrix matrix) {
	for (Vector v : matrix) {
		std::cout << "-";
		PrintNeurons(v);
	}
}
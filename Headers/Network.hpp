#ifndef NETWORK_CPP
#define NETWORK_CPP


#include "Functions.hpp"
#include <vector>
#include <cstdlib>
#include <time.h>

// Debug
#include <iostream>

#define BIAS_ADJUST_DIVISOR 2.f
#define WEIGHT_ADJUST_DIVISOR 5.f
#define NEURON_ADJUST_DIVISOR 5.f

typedef std::vector<float> Vector;
typedef std::vector<std::vector<float>> Matrix;


// The base network class which holds the structure of any supported network
// Also contains essential network methods
class Network {
protected:
	// New
	// This base network class is an ANN structure to allow most other types of 
	// networks to be created with only small adjustments

	// Hidden values
	std::vector<Vector> values;
	// Biases corresponding to each value
	std::vector<Vector> biases;
	// Weights between each layer
	std::vector<Matrix> weights;


	// Initialize values, biases, and weights with random values
	void randomizeNetwork(std::vector<int> neuronCounts);

	Vector calculateLayer(Vector vector, Matrix matrix, Vector biases);
	

public:

	Network(std::vector<int> neuronCounts);
	
	// Propagates through the network
	// Returns the output
	Vector perform(Vector input);

	// Backpropagates through the network's current state
	void train(Vector expectedOutput);
	// Propagates through the network with a given input, 
	// then backpropagates with a given expected output
	void train(Vector input, Vector expectedOutput);

	static Vector calculateCost(Vector actual, Vector expected);
	static void printVector(Vector vec);
	static void printMatrix(Matrix matrix);
};

#endif // NETWORK_HPP
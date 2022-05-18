#ifndef NETWORK_CPP
#define NETWORK_CPP


#include "Functions.hpp"
#include <vector>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <fstream>

#define BIAS_ADJUST_DIVISOR 3.f
#define WEIGHT_ADJUST_DIVISOR 3.f
#define NEURON_ADJUST_DIVISOR 10.f

typedef std::vector<float> Vector;
typedef std::vector<std::vector<float>> Matrix;


// The base network class which holds the structure of any supported network
// Also contains essential network methods
class Network {
protected:
	// New
	// This base network class is an ANN structure to allow most other types of 
	// networks to be created with only small adjustments
	int layerCount;

	


	// Initialize values, biases, and weights with random values
	void randomizeNetwork(std::vector<int> neuronCounts);
	void loadNetwork(std::vector<int> neuronCounts, std::string filename);

	Vector calculateLayer(Vector vector, Matrix matrix, Vector biases);
	

public:

// Hidden values
	std::vector<Vector> values;
	// Biases corresponding to each value
	std::vector<Vector> biases;
	// Weights between each layer
	std::vector<Matrix> weights;


	Network(std::vector<int> neuronCounts);

	// Load the network from a file
	Network(std::vector<int> neuronCounts, std::string filename);
	
	// Propagates through the network
	// Returns the output
	Vector perform(Vector input);

	// Backpropagates through the network's current state
	void train(Vector expectedOutput);
	// Propagates through the network with a given input, 
	// then backpropagates with a given expected output
	void train(Vector input, Vector expectedOutput);


	// Save the network to a file
	void save(std::string filename);


	// print the network to stdout
	void print();

	static Vector calculateCost(Vector actual, Vector expected);
	static void printVector(Vector vec);
	static void printMatrix(Matrix matrix);
};

#endif // NETWORK_HPP
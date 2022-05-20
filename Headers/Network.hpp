#ifndef NETWORK_CPP
#define NETWORK_CPP

#include <vector>
#include <cstdlib>
#include <time.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ios>
#include <string>
#include <thread>
#include "Functions.hpp"

#define BIAS_ADJUST_DIVISOR 50.f
#define WEIGHT_ADJUST_DIVISOR 60.f
#define NEURON_ADJUST_DIVISOR 50.f

typedef std::vector<float> Vector;
typedef std::vector<std::vector<float>> Matrix;


// The base network class which holds the structure of any supported network
// Also contains essential network methods
class Network {
protected:
	// New
	// This base network class is an ANN structure to allow most other types of 
	// networks to be created with only small adjustments
	int layerCount = 2;

	// Hidden values
	std::vector<Vector> values;
	// Biases corresponding to each value
	std::vector<Vector> biases;
	// Weights between each layer
	std::vector<Matrix> weights;


	// Initialize values, biases, and weights with random values
	void randomizeNetwork(std::vector<int> neuronCounts);
	void loadNetwork(std::vector<int> neuronCounts, std::string filename);

	Vector calculateLayer(Vector vector, Matrix matrix, Vector biases);

	// Multiprocessing stuff
	int batchSize = 1;
	int trainingSamples = 1;
	bool writing = false;

	// Returns false if thread is already writing
	bool lock();
	// Frees write access
	void unlock();

	// Backpropagates through the network's current state
	void train(Vector expectedOutput);
	// Propagates through the network with a given input, 
	// then backpropagates with a given expected output
	// void train(Vector input, Vector expectedOutput);

	// Propagates through a given network copy, to be used for multithreading
	void performBackend(Vector input, Network* thisCopy);


public:
	Network(std::vector<int> neuronCounts);

	// Load the network from a file
	Network(std::vector<int> neuronCounts, std::string filename);
	
	// Propagates through the network
	// Returns the output
	Vector perform(Vector input);

	// Backpropagates through the network same as train, but using multithreading
	void batch(Matrix input, Matrix expectedOutput);


	// Save the network to a file
	void save(std::string filename);


	// print the network to stdout
	void print();

	// Set the batch size and training samples to use when training batches
	void prepareBatches(int batchSize, int trainingSamples);

	static Vector calculateCost(Vector actual, Vector expected);
	static void printVector(Vector vec);
	static void printMatrix(Matrix matrix);
};

#endif // NETWORK_HPP
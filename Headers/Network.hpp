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

#define e 2.71828182846

#define BIAS_ADJUST_DIVISOR 50.f
#define WEIGHT_ADJUST_DIVISOR 60.f
#define NEURON_ADJUST_DIVISOR 50.f

typedef std::vector<float> Vector;
typedef std::vector<std::vector<float>> Matrix;


// The base network class which holds the structure of any supported network
// Also contains essential network methods
class Network {
protected:
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

	virtual Vector calculateLayer(Vector vector, Matrix matrix, Vector biases) = 0;
	
	// Returns 2d array of length 2, height and width respectivly
	virtual std::vector<int> getWeightSize(int layer) = 0;

	// Multiprocessing stuff
	bool writing = false;
	// Returns false if thread is already writing
	bool lock();
	// Frees write access
	void unlock();

	// Propagates through a given network copy, to be used for multithreading
	virtual void performBackend(Vector input, Network* thisCopy) = 0;


public:
	// Randomize network
	Network(std::vector<int> neuronCounts);
	// Load the network from a file
	Network(std::vector<int> neuronCounts, std::string filename);
	// Save the network to a file
	void save(std::string filename);

	// Propagates through the network
	// Returns the output
	Vector perform(Vector input);
	// Backpropagates through the network using multithreading
	// Batch size should pretty much always be the number of virtual processors in the machine
	virtual void batch(Matrix input, Matrix expectedOutput, int trainingCycles = 1, int batchSize = 1) = 0;

	// print the network to stdout
	void print();


	// Just crap getters and setters cause I have to
	// And also cause integrating the values into your own projects has obvious benefits
	std::vector<Vector> getValues();
	std::vector<Vector> getBiases();
	std::vector<Matrix> getWeights();

	// Setters
	void setValues(int layer, Vector values);
	void setValue(int layer, int index, float value);

	void setBiases(int layer, Vector biases);
	void setBias(int layer, int index, float value);

	void setWeights(int layer, Matrix weights);
	void setWeights(int layer, int row, Vector weights);
	void setWeight(int layer, int row, int index, float weight);
	
	// Static methods for general functions
	static Vector calculateCost(Vector actual, Vector expected);
	static void printVector(Vector vec);
	static void printMatrix(Matrix matrix);
	static float ReLU(float x);
	static float getRandom(float low, float high);
};

#endif // NETWORK_HPP
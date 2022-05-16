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


class Network {
protected:
	int inputCount;
	int floatingCount;
	int outputCount;

	// Rows := floatingCount
	// Columns := floatingCount + inputCount
	// The first "Columns" columns will be multiplied by the floating neurons
	// you are trying to calculate
	Matrix inputWeights;
	// Rows := outputCount
	// Width := floatingCount + inputCount
	Matrix floatingWeights;

	// list of floating neurons' values
	Vector floatingValues;

	// Vectors of biases
	Vector floatingBiases;
	Vector outputBiases;


	void setupWeights(Matrix* weights, int rows, int cols);
	void setupBiases(Vector* biases, int count);
	void setupInputWeights();
	void setupFloatingWeights();
	void setupFloatingValues();

	Vector findNextLayer(Vector vector, Matrix matrix, Vector biases);

public:

	Network(int inputCount, int floatingCount, int outputCount);
	
	// Calculates the network output given some input value(s)
	Vector prompt(Vector input);

	static Vector calculateCost(Vector actual, Vector expected);
	static void printVector(Vector vec);
	static void printMatrix(Matrix matrix);
};

#endif // NETWORK_HPP
#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "Layer.hpp"
#include "Functions.hpp"
#include <vector>
#include <cstdlib>
#include <time.h>

// Debug
#include <iostream>

typedef std::vector<float> Vector;
typedef std::vector<std::vector<float>> Matrix;


class Network {
private:
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

	Vector train(Vector input, Vector expectedOutput);
};

#endif // NETWORK_HPP
#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "Layer.hpp"
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
private:
	int inputCount;
	int floatingCount;
	int outputCount;

	
	void setupWeights(Matrix* weights, int rows, int cols);
	void setupBiases(Vector* biases, int count);
	void setupInputWeights();
	void setupFloatingWeights();
	void setupFloatingValues();

	Vector findNextLayer(Vector vector, Matrix matrix, Vector biases);

	// Backprop
	void adjustBiases(Vector* biases, Vector* cost);
	void adjustWeights(Matrix* weights, Vector* cost, Vector* values);
	Vector adjustNeurons(Matrix* weights, Vector* cost, Vector* values);

public:
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


	Network(int inputCount, int floatingCount, int outputCount);
	
	// Calculates the network output given some input value(s)
	Vector prompt(Vector input);

	Vector train(Vector input, Vector expectedOutput);
	Vector evolve(Vector);

	static Vector calculateCost(Vector actual, Vector expected);
	static void printVector(Vector vec);
	static void printMatrix(Matrix matrix);
};

#endif // NETWORK_HPP
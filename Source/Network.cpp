#include "../Headers/Network.hpp"
#include <cmath>

Network::Network(int inputCount, int floatingCount, int outputCount) :
	inputCount(inputCount),
	floatingCount(floatingCount),
	outputCount(outputCount) 
	{

	srand(time(NULL));

	// Randomize the weights 
	this->setupInputWeights();
	this->setupFloatingWeights();

	// Randomize the values
	this->setupFloatingValues();

	// Setup the biases
	this->setupBiases(&this->floatingBiases, this->floatingCount);
	this->setupBiases(&this->outputBiases, this->outputCount);

	
}

void Network::setupInputWeights() {
	this->setupWeights(&this->inputWeights, this->floatingCount, this->floatingCount + this->inputCount);
}

void Network::setupFloatingWeights() {
	this->setupWeights(&this->floatingWeights, this->outputCount, this->floatingCount + this->inputCount);
}

void Network::setupWeights(Matrix* weights, int rows, int cols) {
	// Clear the weights
	weights->clear();

	// Setup the matrix by [rows][columns]
	for (int r=0;r<rows;r++) {
		// Add a new row to the matrix
		std::vector<float> tmpRow;
		weights->push_back(std::vector<float>());
		
		auto row = &(*weights)[r];

		for (int c=0;c<cols;c++) {
			// Generate a random weight between -4.f and 4.f
			// with up to 6 decimal points of accuracy 
			row->push_back(getRandom());
		}
	}
}

void Network::setupBiases(Vector* biases, int count) {
	// Clear the biases
	biases->clear();

	for (int r=0;r<count;r++) {
		// Biases will all generate with a default value of 0
		biases->push_back(0);
	}
}

void Network::setupFloatingValues() {
	// Clear the values
	this->floatingValues.clear();

	// Generate a random number between -1 and 1 for the floating neurons
	for (int i=0;i<this->floatingCount;i++) {
		this->floatingValues.push_back(getRandom(-1, 1));
	}
}

// Multiply two matrices together
// Works even if one or more is a vector (still has to be a 2d vector, 
// but there is just one element in the second dimension)
Vector Network::findNextLayer(Vector vector, Matrix matrix, Vector biases) {
	if (vector.size() != matrix[0].size())
		std::cerr << "findNextLayer-Network.cpp: Vector and matrix are incompatible, size mismatch." << std::endl;

	Vector result;
	// The size of matrix is the number of rows, which is also
	// the number of rows of the output vector
	for (int r=0;r< matrix.size() ;r++) {
		float sum = 0;

		// Loop through all the "input" neurons and multiply by the weights matrix columns
		for (int i=0;i< vector.size() ;i++) {
			sum += vector[i] * matrix[r][i];
		}

		// When multiplying matrices, all the columns are multiplied by the
		// rows of the input vector, then those products are added together to get
		// the row of the output vector. so now we need to put that sum in the 
		// next row of the result

		// before we do that though, the resulting value must be constrained
		sum = ReLU(sum);

		// Finally to add the bias corrosponding to this neuron
		sum += biases[r];

		// Next we set the "sum" as the row of the vector
		result.push_back(sum);
	}

	return result;
}

Vector Network::prompt(Vector input) {
	// concat the input values to the end of the floating values matrix.
	// This matrix just gives the output values, so since we want to
	// relate the input directly to the output in addition to through neuron paths...
	Vector floatingAndInput = this->floatingValues;
	for (float x : input)
		floatingAndInput.push_back(x);

	// Calculate the floating layer
	this->floatingValues = findNextLayer(floatingAndInput, this->inputWeights, this->floatingBiases);

	// Calculate output
	Vector output = findNextLayer(floatingAndInput, this->floatingWeights, this->outputBiases);

	return output;
}

Vector Network::train(Vector input, Vector expectedOutput) {
	// concat the input values to the end of the floating values matrix.
	// This matrix just gives the output values, so since we want to
	// relate the input directly to the output in addition to through neuron paths...
	Vector floatingAndInput = this->floatingValues;
	for (float x : input)
		floatingAndInput.push_back(x);

	// Calculate the floating layer
	this->floatingValues = findNextLayer(floatingAndInput, this->inputWeights, this->floatingBiases);
	
	// Calculate output
	Vector output = findNextLayer(floatingAndInput, this->floatingWeights, this->outputBiases);

	// Step 1: Adjust Biases
	//for (int i=0;i)

	// Step 2: Adjust Weights
	
	// Step 3: Adjust Neurons


	// Return the output in case the user wants to use it
	return output;
}
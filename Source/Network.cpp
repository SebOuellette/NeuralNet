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

// Backpropagation method
Vector Network::train(Vector input, Vector expectedOutput) {
	// concat the input values to the end of the floating values matrix.
	// This matrix just gives the output values, so since we want to
	// relate the input directly to the output in addition to through neuron paths...
	Vector floatingAndInput = input;
	for (float x : this->floatingValues)
		floatingAndInput.push_back(x);

	// Calculate the floating layer
	this->floatingValues = findNextLayer(floatingAndInput, this->inputWeights, this->floatingBiases);
	
	// Calculate output
	Vector output = findNextLayer(floatingAndInput, this->floatingWeights, this->outputBiases);



	// Adjust the connections to the output neurons
	Vector cost = calculateCost(output, expectedOutput);
	// Step 1: Adjust Biases for Output Neurons
	adjustBiases(&this->outputBiases, &cost);

	// Step 2: Adjust Weights before Output Neurons
	adjustWeights(&this->floatingWeights, &cost, &floatingAndInput);
	
	// Step 3: Adjust Neurons
	Vector floatingExpected = adjustNeurons(&this->floatingWeights, &cost, &this->floatingValues);
	


	// // Adjust the connections to the floating neurons
	// cost = calculateCost(this->floatingValues, floatingExpected);
	// // Step 1: Adjust Biases for Floating Neurons
	// adjustBiases(&this->floatingBiases, &cost);

	// // Step 2: Adjust Weights before Floating Neurons
	// adjustWeights(&this->inputWeights, &cost, &floatingAndInput);


	// Return the output in case the user wants to use it
	return output;
}

void Network::adjustBiases(Vector* biases, Vector* cost) {
	for (int i=0;i<this->outputCount;i++) {
		this->outputBiases[i] += (*cost)[i] / BIAS_ADJUST_DIVISOR;
	}
}

void Network::adjustWeights(Matrix* weights, Vector* cost, Vector* values) {
	for (int r=0;r<weights->size();r++) { // For each output neuron
		
		for (int n=0;n<values->size();n++) { // For each input neuron / connection to selected output neuron
			// The wcost is the difference between the cost and the value.
			float wcost = (*cost)[r] - (*values)[n];
			(*weights)[r][n] += wcost / WEIGHT_ADJUST_DIVISOR;
		}
	}
}

Vector Network::adjustNeurons(Matrix* weights, Vector* cost, Vector* values) {
	// The list of adjustments to apply to 
	Vector neuronAdjustments(values->size());

	for (int r=0;r<weights->size();r++) { // For each output neuron
		
		for (int n=0;n<values->size();n++) { // For each input neuron / connection to selected output neuron
			// The wcost is the difference between the cost and the value.
			float wcost = (*cost)[r] - (*values)[n];
			neuronAdjustments[n] += wcost / NEURON_ADJUST_DIVISOR;
		}
	}

	return neuronAdjustments;
}

Vector Network::calculateCost(Vector actual, Vector expected) {
	// The vector containing the cost of the output
	Vector cost(actual.size());

	for (int i=0;i<actual.size();i++)
		cost[i] = expected[i] - actual[i];

	return cost;
}

void Network::printVector(Vector vec) {
	std::cout << "[";
	
	for(float val : vec)
		std::cout << val << ", ";

	std::cout << "\b\b]" << std::endl;
}

void Network::printMatrix(Matrix matrix) {
	for (Vector r : matrix) {
		std::cout << "[";

		for (float val : r)
			std::cout << val << ", ";

		std::cout << "\b\b]" << std::endl;
	}

}
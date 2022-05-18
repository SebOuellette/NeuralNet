#include "../Headers/Network.hpp"
#include <cmath>

Network::Network(std::vector<int> neuronCounts) {

	srand(time(NULL));

	this->randomizeNetwork(neuronCounts);
}

void Network::randomizeNetwork(std::vector<int> neuronCounts) {
	int layerCount = neuronCounts.size();
	
	// Fill values with floats 0. to 1.
	for (int i=0;i<layerCount - 2;i++) {
		Vector layer;
		for (int l=0;l<neuronCounts[i + 1];l++) {
			layer.push_back(getRandom(0., 1.));
		}
		this->biases.push_back(layer);
	}

	// Fill biases with 0
	for (int i=0;i<layerCount - 1;i++) {
		Vector layer;
		for (int l=0;l<neuronCounts[i + 1];l++) {
			layer.push_back(getRandom(0., 1.));
		}
		this->biases.push_back(layer);
	}

	// Fill weights with floats -4. to 4.
	for (int i=0;i<layerCount - 1;i++) {
		int matHeight = neuronCounts[i];
		int matWidth = neuronCounts[i + 1];

		// Loop through all the rows and columns for this weight matrix
		Matrix weightsData;
		for (int row=0;row<matHeight;row++) {
			Vector rowData(0);
			for (int col=0;col<matWidth;col++) {
				// Add a random data to the coordinate
				rowData.push_back(getRandom(-4., 4.));
			}
			weightsData.push_back(rowData);
		}
		this->weights.push_back(weightsData);
	}
}

// Multiply two matrices together
// Works even if one or more is a vector (still has to be a 2d vector, 
// but there is just one element in the second dimension)
Vector Network::calculateLayer(Vector vector, Matrix matrix, Vector biases) {
	if (vector.size() != matrix[0].size())
		std::cerr << "calculateLayer-Network.cpp: Vector and matrix are incompatible, size mismatch."
		 << "Vector: " << vector.size() << " Matrix: " << matrix[0].size() << std::endl;

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

		// Add the bias corrosponding to this neuron
		sum += biases[r];

		// Finally, the resulting value must be constrained
		sum = ReLU(sum);

		// Next we set the "sum" as the row of the vector
		result.push_back(sum);
	}

	return result;
}

// Propagate through the network
Vector Network::perform(Vector input) {
	this->values[0] = input; // just to make it easy

	int layerCount = this->values.size();

	// Propagate through the network
	for (int l=0;l<layerCount - 1;l++) {
		this->values[l+1] = calculateLayer(this->values[l], this->weights[l], this->biases[l]);
	}

	return this->values.back();
}

void Network::train(Vector input, Vector expectedOutput) {
	perform(input);

	// Don't have to return, but if I need to change the return type later...
	return train(expectedOutput);
}


// Static methods
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
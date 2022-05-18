#include "../Headers/Network.hpp"
#include <thread>

Network::Network(std::vector<int> neuronCounts) {
	if (neuronCounts.size() < 2) {
		std::cerr << "Network must have at least 2 layers given in constructor" << std::endl;
		exit(1);
	}

	srand(time(NULL));

	this->randomizeNetwork(neuronCounts);
}

Network::Network(std::vector<int> neuronCounts, std::string filename) {
	if (neuronCounts.size() < 2) {
		std::cerr << "Network must have at least 2 layers given in constructor" << std::endl;
		exit(1);
	}

	this->loadNetwork(neuronCounts, filename);
}

void Network::loadNetwork(std::vector<int> neuronCounts, std::string filename) {
	std::ifstream file(filename);

	if (!file) {
		std::cerr << filename << " failed to open" << std::endl;
		exit(1);
	}

	// if I need to throw crap out, here's where I throw it
	char tmpBuffer;
	float numBuffer;

	// And gotta initialize this
	this->layerCount = neuronCounts.size();

	// Remember the input neurons, hidden neurons, and output neurons are all saved
	// Values
	for (int i=0;i<neuronCounts.size();i++) {
		Vector values;
		file >> tmpBuffer; // [

		for (int l=0;l<neuronCounts[i];l++) {
			file >> numBuffer; // value
			file >> tmpBuffer; // ,

			// Do something with the value
			values.push_back(numBuffer);
		}
		file >> tmpBuffer; // ]

		// Add the local values to the stored values
		this->values.push_back(values);
	}

	// \n

	// Biases
	for (int i=0;i<neuronCounts.size() - 1;i++) {
		Vector biases;
		file >> tmpBuffer; // [

		for (int l=0;l<neuronCounts[i + 1];l++) {
			file >> numBuffer; // bias
			file >> tmpBuffer; // ,

			// Do something with the bias
			biases.push_back(numBuffer);
		}
		file >> tmpBuffer; // ]

		// Add the local values to the stored values
		this->biases.push_back(biases);
	}

	// \n

	// Weights
	for (int i=0;i<neuronCounts.size() - 1;i++) {
		Matrix layerWeights;
		file >> tmpBuffer; // [

		// rows = output size
		for (int r=0;r<neuronCounts[i + 1];r++) {
			Vector row;
			file >> tmpBuffer; // [

			// cols = input size
			for (int c=0;c<neuronCounts[i];c++) { // haha c++
				file >> numBuffer; // weight
				file >> tmpBuffer; // ,

				// Do something with the weight
				row.push_back(numBuffer);
			}
			file >> tmpBuffer; // ]

			layerWeights.push_back(row);
		}

		file >> tmpBuffer; // [
		this->weights.push_back(layerWeights);
	}

	
	file.close();
}

void Network::save(std::string filename) {
	std::ofstream file(filename);

	if (!file) {
		std::cerr << filename << " failed to open" << std::endl;
		exit(1);
	}

	// I don't want to make so many small writes, so I'll have a buffer
	std::string writeBuffer = "";

	// Remember the input neurons, hidden neurons, and output neurons are all saved
	// Values
	for (int i=0;i<this->values.size();i++) {
		writeBuffer += '[';
		for (int l=0;l<this->values[i].size();l++) {
			writeBuffer += std::to_string(this->values[i][l]) +',';
		}
		writeBuffer += ']';
	}

	writeBuffer += '\n';

	// Biases
	for (int i=0;i<this->biases.size();i++) {
		writeBuffer += '[';
		for (int l=0;l<this->biases[i].size();l++) {
			writeBuffer += std::to_string(this->biases[i][l]) + ',';
		}
		writeBuffer += ']';
	}
	
	writeBuffer += '\n';

	// Weights
	for (int i=0;i<this->weights.size();i++) {
		writeBuffer += '[';
		for (int r=0;r<this->weights[i].size();r++) {
			writeBuffer += '[';
			for (int c=0;c<this->weights[i][r].size();c++)
				writeBuffer += std::to_string(this->weights[i][r][c]) + ',';
			writeBuffer += ']';
		}
		writeBuffer += ']';
	}

	// Finally overwrite the file with the writeBuffer
	file << writeBuffer;

	file.close();
}

void Network::randomizeNetwork(std::vector<int> neuronCounts) {
	// Empty the network
	this->values = std::vector<Vector>(0);
	this->biases = std::vector<Vector>(0);
	this->weights = std::vector<Matrix>(0);

	this->layerCount = neuronCounts.size();
	
	// Fill values with floats 0. to 1.
	for (int i=0;i<layerCount;i++) {
		Vector layer;
		for (int l=0;l<neuronCounts[i];l++) {
			layer.push_back(getRandom(0., 1.));
		}
		this->values.push_back(layer);
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
		int matHeight = neuronCounts[i + 1];
		int matWidth = neuronCounts[i];

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
	if (vector.size() != matrix[0].size()) {
		std::cerr << "calculateLayer-Network.cpp: Vector and matrix are incompatible, size mismatch."
		 << "Vector: " << vector.size() << " Matrix: " << matrix[0].size() << std::endl;
		exit(1);
	}

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

void Network::prepareBatches(int batchSize, int trainingSamples) {
	this->batchSize = batchSize;
	this->trainingSamples = trainingSamples;
}

void Network::batch(Vector input, Vector expectedOutput) {
	if (this->values.back().size() != expectedOutput.size()) {
		std::cerr << "Network training: expected output not the same size as actual output" << std::endl;
		exit(1);
	}

	this->perform(input);

	this->batch(expectedOutput);
}

void Network::batch(Vector expectedOutput) {
	if (this->values.back().size() != expectedOutput.size()) {
		std::cerr << "Network training: expected output not the same size as actual output" << std::endl;
		exit(1);
	}

	std::vector<std::thread*> threads;

	for(int t=0;t<this->batchSize;t++) {
		threads.push_back(new std::thread([&] {
			for (int trainingCycle=0;trainingCycle<this->trainingSamples;trainingCycle++) {

				Vector expected = expectedOutput;

				// Loop through all layers (recursion but without the downsides)
				int layer = this->layerCount;
				while (layer-- > 1) {

					Vector* lastValues = &this->values[layer - 1];
					Vector* biases = &this->biases[layer - 1];
					Matrix* weights = &this->weights[layer - 1];

					// For Step 3
					Vector neuronAdjustments(lastValues->size());

					Vector cost = calculateCost(this->values[layer], expected);

					// Train the connections to each output neuron
					for (int r=0;r<this->values[layer].size();r++) {
						
						// Step 1: Adjust Bias
						(*biases)[r] += cost[r] / BIAS_ADJUST_DIVISOR;
						
						// Step 2: Adjust weights
						for (int l=0;l<lastValues->size();l++) {
							(*weights)[r][l] += cost[r] * (*lastValues)[l] / WEIGHT_ADJUST_DIVISOR;
						}

						// Step 3: Adjust neurons
						for (int l=0;l<lastValues->size();l++) {
							neuronAdjustments[l] += cost[r] * (*lastValues)[l] / NEURON_ADJUST_DIVISOR;
						}
					}

					// // Continuing Step 3: Create the new expected value for the layer before
					expected.resize(lastValues->size());
					for (int i=0;i<expected.size();i++) {
						expected[i] = (*lastValues)[i] + neuronAdjustments[i];
					}
				}
			}

		}));
	}

	// Join the threads
	for (int i=0;i<batchSize;i++) {
		threads[i]->join();
		delete threads[i];
	}
}

void Network::train(Vector expectedOutput) {
	if (this->values.back().size() != expectedOutput.size()) {
		std::cerr << "Network training: expected output not the same size as actual output" << std::endl;
		exit(1);
	}

	
}

void Network::print() {
	for (int i=0;i<=this->weights.size();i++) {
		std::cout << "Layer " << i << " - ";
		Network::printVector(this->values[i]);
		std::cout << std::endl;
		

		if (i < this->weights.size()) {
			Network::printMatrix(this->weights[i]);
			std::cout << std::endl;

			Network::printVector(this->biases[i]);
			std::cout << std::endl;
		}
	}
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
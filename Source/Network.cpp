#include "../Headers/Network.hpp"

Network::Network(std::vector<int> neuronCounts) {
	if (neuronCounts.size() < 2) {
		std::cerr << "Network must have at least 2 layers given in constructor" << std::endl;
		exit(1);
	}

	srand(time(NULL));

	// Cannot randomize network here because it depends on child class's getWeightSize override
}

Network::Network(std::vector<int> neuronCounts, std::string filename) {
	if (neuronCounts.size() < 2) {
		std::cerr << "Network must have at least 2 layers given in constructor" << std::endl;
		exit(1);
	}

	// Cannot load network here because it depends on child class's getWeightSize override
}

// Copy constructor
Network::Network(Network* network) {
	this->layerCount = network->layerCount;
	this->values = network->getValues();
	this->biases = network->getBiases();
	this->weights = network->getWeights();
}

Vector Network::perform(Vector input) {
	this->performBackend(input, this);

	return this->values.back();
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
		std::vector<int> weightSize = this->getWeightSize(i);

		// Loop through all the rows and columns for this weight matrix
		Matrix weightsData;
		for (int row=0;row<weightSize[0];row++) {
			Vector rowData(0);
			for (int col=0;col<weightSize[1];col++) {
				// Add a random data to the coordinate
				rowData.push_back(getRandom(-4., 4.));
			}
			weightsData.push_back(rowData);
		}
		this->weights.push_back(weightsData);
	}
}

void Network::loadNetwork(std::vector<int> neuronCounts, std::string filename) {
	std::ifstream file(filename);

	// If the savefile wasn't found, randomize the network instead
	if (!file) {
		std::cerr << filename << " failed to open, randomizing network" << std::endl;
		
		this->randomizeNetwork(neuronCounts);

		return;
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
		std::vector<int> weightSize = this->getWeightSize(i);

		Matrix layerWeights;
		file >> tmpBuffer; // [

		// rows = output size
		for (int r=0;r<weightSize[0];r++) {
			Vector row;
			file >> tmpBuffer; // [

			// cols = input size
			for (int c=0;c<weightSize[1];c++) { // haha c++
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

// Multiply a provided input by a given weight matrix, and add biases to produce an output vector
Vector Network::calculateLayer(Vector vector, Matrix matrix, Vector biases) {
	if (vector.size() > matrix[0].size()) {
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
		std::vector<int> weightSize = this->getWeightSize(i);

		writeBuffer += '[';
		for (int r=0;r<weightSize[0];r++) {
			writeBuffer += '[';
			for (int c=0;c<weightSize[1];c++)
				writeBuffer += std::to_string(this->weights[i][r][c]) + ',';
			writeBuffer += ']';
		}
		writeBuffer += ']';
	}

	// Finally overwrite the file with the writeBuffer
	file << writeBuffer;

	file.close();
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

bool Network::lock() {
	// Poll to check if no other thread is writing
	if (this->writing) return false;

	// grab write access
	this->writing = true;
	return true;
}

void Network::unlock() {
	// Free write access
	this->writing = false;
}

// Getters
std::vector<Vector> Network::getValues() {
	return this->values;
}
std::vector<Vector> Network::getBiases() {
	return this->biases;
}
std::vector<Matrix> Network::getWeights() {
	return this->weights;
}

// Setters
// Values
void Network::setValues(int layer, Vector values) {
	this->values[layer] = values;
}
void Network::setValue(int layer, int index, float value) {
	this->values[layer][index] = value;
}
// Biases
void Network::setBiases(int layer, Vector biases) {
	this->biases[layer] = biases;
}
void Network::setBias(int layer, int index, float value) {
	this->biases[layer][index] = value;
}
// Weights
void Network::setWeights(int layer, Matrix weights) {
	this->weights[layer] = weights;
}
void Network::setWeights(int layer, int row, Vector weights) {
	this->weights[layer][row] = weights;
}
void Network::setWeight(int layer, int row, int index, float weight) {
	this->weights[layer][row][index] = weight;
}

// Static methods
Vector Network::calculateCost(Vector actual, Vector expected) {
	if (actual.size() != expected.size()) {
		std::cerr << "Network::calculateCost - actual not the same size as expected" << std::endl;
		exit(1);
	}
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

// Logistic curve // Smooth 0. - 1.
float Network::ReLU(float x) {
	return 1.f / (1.f + powf(e, -x));
}

// Generate a random float between -4 and 4
//		-4 and 4 because these values are used for weights
// 		Arguments accepted to change the default values
float Network::getRandom(float low, float high) {
	return std::fmod((float)rand() / 1000000, high - low) + low;
}
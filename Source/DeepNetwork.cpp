#include "../Headers/DeepNetwork.hpp"

Instance::Instance(int inputCount, int floatingCount, int outputCount) :
	Network(inputCount, floatingCount, outputCount) {}

void Instance::mutate() {
	// Generate mutation count
	auto mutations = rand() % (MAX_MUTATIONS - MIN_MUTATIONS) + MIN_MUTATIONS;

	// Mutate
	for (int i=0;i<mutations;i++) {
		int variableToModify = rand() % 5;
		int row, col;

		// 50/50 add/subtract
		float adjustment = MUTATION_ADJUSTMENT * (rand() % 2) == 0 ? 1 : -1;

		switch(variableToModify) {
			case 0: // inputWeights
				row = rand() % this->inputWeights.size();
				col = rand() % this->inputWeights[0].size();

				this->inputWeights[row][col] += adjustment;
				break;
			case 1: // floatingWeights
				row = rand() % this->floatingWeights.size();
				col = rand() % this->floatingWeights[0].size();

				this->floatingWeights[row][col] += adjustment;
				break;
			case 2: // floatingValues
				row = rand() % this->floatingValues.size();

				this->floatingValues[row] += adjustment;
				break;
			case 3: // floatingBiases
				row = rand() % this->floatingBiases.size();

				this->floatingBiases[row] += adjustment;
				break;
			case 4: // outputBiases
				row = rand() % this->outputBiases.size();

				this->outputBiases[row] += adjustment;
				break;
			default:
				break;
		}
	}
}


DeepNetwork::DeepNetwork(int inputCount, int floatingCount, int outputCount, int instanceCount) :
	inputCount(inputCount),
	floatingCount(floatingCount),
	outputCount(outputCount),
	instanceCount(instanceCount) {
		
		for (int i=0;i<instanceCount;i++) {
			Instance newInstance(inputCount, floatingCount, outputCount);
			
			this->instances.push_back(newInstance);
		}
}

std::vector<Instance>* DeepNetwork::getInstances() {
	return &this->instances;
}

void DeepNetwork::propagate() {
	this->instances = evolveGen();
}

std::vector<Instance> DeepNetwork::evolveGen() {
	std::vector<Instance> newInstances;

	for (int i=0;i<this->instanceCount;i++) {
		Instance* randomInstance = &this->instances[rand() % this->instanceCount];
		
		newInstances.push_back(*randomInstance);
		newInstances[i].mutate();
	}

	return newInstances;
}
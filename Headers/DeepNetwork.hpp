#ifndef DEEP_NETWORK_CPP
#define DEEP_NETWORK_CPP

#include "Network.hpp"

#define MIN_MUTATIONS 0
#define MAX_MUTATIONS 2
#define MUTATION_ADJUSTMENT 0.001


class Instance : public Network {
private:
public:
	Instance(int inputCount, int floatingCount, int outputCount);
	void mutate();
};

class DeepNetwork {
private:
	int inputCount;
	int floatingCount;
	int outputCount;
	int instanceCount;

	std::vector<Instance> instances;
	
public:

	DeepNetwork(int inputCount, int floatingCount, int outputCount, int instanceCount);
	
	// Returns 
	Vector prompts(Vector input);

	std::vector<Instance>* getInstances();

	void propagate();
	std::vector<Instance> evolveGen();
};

#endif // NETWORK_HPP
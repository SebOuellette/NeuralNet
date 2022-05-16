#ifndef DEEP_NETWORK_CPP
#define DEEP_NETWORK_CPP

#include "Network.hpp"

class Instance : public Network {
private:
public:
	Instance(int inputCount, int floatingCount, int outputCount);
	void mutate(int chanceOfMutation);
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
	
	// Calculates the network output given some input value(s)
	Vector prompt(Vector input);

	Vector evolve(Vector);
};

#endif // NETWORK_HPP
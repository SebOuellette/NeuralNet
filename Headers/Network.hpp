#ifndef NETWORK_CPP
#define NETWORK_CPP

#include <vector>
#include "Neuron.hpp"

typedef std::vector<Neuron> neuronList;

class Network {
private:
	neuronList inputNeurons;
	neuronList floatingNeurons;
	neuronList outputNeurons;

	std::vector<float> uniformOutputs;
	
public:
	Network(int inputNeuronCount, int floatingNeuronCount, int outputNeuronCount);
	
	void askNetwork();
	static void PrintNeurons();
};

#endif // NETWORK.CPP
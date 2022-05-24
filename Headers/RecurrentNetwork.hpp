#ifndef RECURRENT_NETWORK_HPP
#define RECURRENT_NETWORK_HPP

#include "Network.hpp"

#define BIAS_ADJUST_DIVISOR 50.f
#define WEIGHT_ADJUST_DIVISOR 60.f
#define NEURON_ADJUST_DIVISOR 50.f

// The Recurrent Network class
class RecurrentNetwork : public Network {
protected:
	int lookbackSize;
	std::vector<Vector> previousOutputs;

	// Propagates through a given network copy, to be used for multithreading
	void performBackend(Vector input, Network* thisCopy) override;
	void performBackend(Vector input, RecurrentNetwork* thisCopy);

	std::vector<int> getWeightSize(int layer) override;


public:
	RecurrentNetwork(std::vector<int> neuronCounts, int lookbackSize = 0);

	// Load the network from a file
	RecurrentNetwork(std::vector<int> neuronCounts, std::string filename, int lookbackSize = 0);

	// Backpropagates through the network using multithreading
	// Batch size should pretty much always be the number of virtual processors in the machine
	void batch(Matrix input, Matrix expectedOutput, int trainingCycles = 1, int batchSize = 1) override;
};

#endif // RECURRENT_NETWORK_HPP
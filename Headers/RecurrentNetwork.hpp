#ifndef RECURRENT_NETWORK_HPP
#define RECURRENT_NETWORK_HPP
#include "Network.hpp"



#include "Network.hpp"

// class RecurrentNetwork : public Network {
// private:
// 	// Initialize values, biases, and weights with random values
// 	void randomizeNetwork(std::vector<int> neuronCounts);
// 	void loadNetwork(std::vector<int> neuronCounts, std::string filename);

// 	Vector calculateLayer(Vector vector, Matrix matrix, Vector biases);

// 	// Multiprocessing
// 	// Propagates through a given network copy, to be used for multithreading
// 	void performBackend(Vector input, Network* thisCopy);
// public:
// 	RecurrentNetwork(std::vector<int> neuronCounts);
// 	RecurrentNetwork(std::vector<int> neuronCounts, std::string filename);

// 	// Save the network to a file
// 	void save(std::string filename);

// 	// Propagates through the network
// 	// Returns the output
// 	Vector perform(Vector input);
// 	// Backpropagates through the network
// 	void batch(Matrix input, Matrix expectedOutput, int trainingCycles = 1, int batchSize = 1);
// };

#endif // RECURRENT_NETWORK_HPP
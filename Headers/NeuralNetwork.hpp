#ifndef NEURAL_NETWORK_CPP
#define NEURAL_NETWORK_CPP

#include "Network.hpp"


class NeuralNetwork : public Network  {
private:
	// Backprop
	void adjustBiases(Vector* biases, Vector* cost);
	void adjustWeights(Matrix* weights, Vector* cost, Vector* values);
	Vector adjustNeurons(Matrix* weights, Vector* cost, Vector* values);

public:
	NeuralNetwork(int inputCount, int floatingCount, int outputCount);
	
	Vector train(Vector input, Vector expectedOutput);
};

#endif // NETWORK_HPP
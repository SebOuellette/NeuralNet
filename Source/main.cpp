#include <iostream>
#include "../Headers/NeuralNetwork.hpp"
#include "../Headers/DeepNetwork.hpp"

#define TRAINING_SAMPLES 100

int main(int argc, char* argv[]) {
	// NeuralNetwork network(3, 3, 3);

	// for (int i=0;i<TRAINING_SAMPLES;i++) {
	// 	network.train({0, 0, 0}, {1, 1, 1});
	// 	network.train({0, 0, 1}, {1, 1, 0});
	// 	network.train({0, 1, 0}, {1, 0, 1});
	// 	network.train({0, 1, 1}, {1, 0, 0});
	// 	network.train({1, 0, 0}, {0, 1, 1});
	// 	network.train({1, 0, 1}, {0, 1, 0});
	// 	network.train({1, 1, 0}, {0, 0, 1});
	// 	network.train({1, 1, 1}, {0, 0, 0});
	// }

	// network.prompt({1, 0, 1});

	// NeuralNetwork::printVector(NeuralNetwork::calculateCost(network.prompt({1, 0, 1}), {0, 1, 0}));




	return 0;
}
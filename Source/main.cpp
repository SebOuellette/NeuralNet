#include <iostream>
#include "../Headers/Network.hpp"

#define TRAINING_SAMPLES 300

int main(int argc, char* argv[]) {
	Network network(2, 3, 2);

	network.prompt({0.1, 0.2});

	for (int i=0;i<10000;i++) {
		network.train({0, 0}, {1, 1});
		network.train({0, 1}, {1, 0});
		network.train({1, 0}, {0, 1});
		network.train({1, 1}, {0, 0});
	}

	Network::printMatrix(network.inputWeights);
	std::cout << std::endl;
	Network::printVector(network.floatingBiases);
	std::cout << std::endl;

	Network::printVector(network.floatingValues);
	std::cout << std::endl;

	Network::printMatrix(network.floatingWeights);
	std::cout << std::endl;
	Network::printVector(network.outputBiases);
	std::cout << std::endl;
	Network::printVector(network.prompt({0, 1}));

	return 0;
}
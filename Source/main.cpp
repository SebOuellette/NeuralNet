#include <iostream>
#include "../Headers/Network.hpp"

int main(int argc, char* argv[]) {
	Network network(2, {3, 3}, 2);
	// Create the network input
	Vector input = {0.2, 0.7};
	std::cout << "Input: ";
	Network::PrintNeurons(input);

	// Declare the expected output
	Vector expected = {0.0, 1.0};
	std::cout << "Expected: ";
	Network::PrintNeurons(expected);

	// Ask the network for ouput, without training
	Vector output = {1.0, 0.0};//network.askNetwork({0.2, 0.7});
	Network::PrintNeurons(output);

	// Calculate the cost of the network
	float cost = Network::calculateNetworkCost({0.0, 1.0}, output);
	std::cout << "Network Cost: " << cost << std::endl;

	// Train the network
	for (int i=0;i<10;i++) {
		Network::PrintNeurons(network.train({0.2, 0.7}, {0.0, 1.0}));
	}

	return 0;
}
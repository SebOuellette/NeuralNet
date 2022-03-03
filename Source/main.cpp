#include <iostream>
#include "../Headers/Network.hpp"

int main(int argc, char* argv[]) {
	// XOR
	Network network(2, {10}, 1);

	// Ask the network for ouput, without training
	std::cout << "Untrained outputs: " << std::endl;
	Network::PrintNeurons(network.askNetwork({0, 0}));
	Network::PrintNeurons(network.askNetwork({0, 1}));
	Network::PrintNeurons(network.askNetwork({1, 0}));
	Network::PrintNeurons(network.askNetwork({1, 1}));

	std::cout << "Untrained cost: " << Network::calculateNetworkCost({1}, network.askNetwork({1, 0})) << std::endl;

	// Train the network
	for (int i=0;i<1;i++) {
		network.train({0, 0}, {0});
		network.train({0, 1}, {0});
		network.train({1, 0}, {0});
		network.train({1, 1}, {0});
	}

	// Ask the network for ouput, after training
	std::cout << "Trained outputs: " << std::endl;
	Network::PrintNeurons(network.askNetwork({0, 0}));
	Network::PrintNeurons(network.askNetwork({0, 1}));
	Network::PrintNeurons(network.askNetwork({1, 0}));
	Network::PrintNeurons(network.askNetwork({1, 1}));

	std::cout << "Trained cost: " << Network::calculateNetworkCost({1}, network.askNetwork({1, 0})) << std::endl;
	
	return 0;
}
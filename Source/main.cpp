#include <iostream>
#include "../Headers/Network.hpp"

#define TRAINING_SAMPLES 1000

int main(int argc, char* argv[]) {
	// XOR
	Network network(3, {10}, 3);

	// Ask the network for ouput, without training
	std::cout << "Untrained outputs: " << std::endl;
	Network::PrintNeurons(network.askNetwork({0, 0, 0}));
	Network::PrintNeurons(network.askNetwork({0, 1, 1}));
	Network::PrintNeurons(network.askNetwork({1, 0, 0}));
	Network::PrintNeurons(network.askNetwork({1, 1, 1}));

	//std::cout << "Untrained cost: " << Network::calculateNetworkCost({1}, network.askNetwork({1, 0})) << std::endl;
	
	std::cout << " -- Debug --" << std::endl;
	// Train the network
	for (int i=0;i<TRAINING_SAMPLES;i++) {
		network.train({0, 0, 0}, {1, 1, 1});
		network.train({0, 0, 1}, {1, 1, 0});
		network.train({0, 1, 0}, {1, 0, 1});
		//network.train({0, 1, 1}, {1, 0, 0});
		network.train({1, 0, 0}, {0, 1, 1});
		network.train({1, 0, 1}, {0, 1, 0});
		network.train({1, 1, 0}, {0, 0, 1});
		network.train({1, 1, 1}, {0, 0, 0});
		std::cout << ((double)i / (double)TRAINING_SAMPLES) * 100 << "%" << std::endl;
		//std::cout << "  ---------\nNEXT - SAMPLE\n  ---------" << std::endl;
	}
	std::cout << " -- End of Debug --" << std::endl;

	// Ask the network for ouput, after training
	std::cout << "Trained outputs: " << std::endl;
	Network::PrintNeurons(network.askNetwork({0, 0, 0}));
	Network::PrintNeurons(network.askNetwork({0, 1, 1}));
	Network::PrintNeurons(network.askNetwork({1, 0, 0}));
	Network::PrintNeurons(network.askNetwork({1, 1, 1}));
	Network::PrintNeurons(network.askNetwork({1, 0, 1}));

	//std::cout << std::endl;
	//network.askNetwork({0, 0});
	//
	//network.display();

	//std::cout << "Trained cost: " << Network::calculateNetworkCost({1}, network.askNetwork({1, 0})) << std::endl;
	
	return 0;
}
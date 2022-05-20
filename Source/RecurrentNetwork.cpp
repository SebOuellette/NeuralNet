#include "../Headers/RecurrentNetwork.hpp"

RecurrentNetwork::RecurrentNetwork(std::vector<int> neuronCounts) :
	Network(neuronCounts) {
		if (neuronCounts.size() < 2 || neuronCounts.size() > 3) {
			std::cerr << "Recurrent networks may only have 2 or 3 layers" << std::endl;
			exit(1);
		}

}

RecurrentNetwork::RecurrentNetwork(std::vector<int> neuronCounts, std::string filename) :
	Network(neuronCounts, filename) {
		if (neuronCounts.size() < 2 || neuronCounts.size() > 3) {
			std::cerr << "Recurrent networks may only have 2 or 3 layers" << std::endl;
			exit(1);
		}

}
#include <iostream>
#include "../Headers/Network.hpp"

int main(int argc, char* argv[]) {
	
	Network network(2, {3, 3}, 2);
	Vector output = network.askNetwork(Vector{0.2, 0.7});
	Network::PrintNeurons(output);

	float cost = Network::calculateNetworkCost(Vector{0.0, 1.0}, output);
	std::cout << "Cost: " << cost << std::endl;

	return 0;
}
#include <iostream>
#include <cmath>
#include <iterator>
//#include "../Headers/NeuralNetwork.hpp"
//#include "../Headers/DeepNetwork.hpp"
#include "../Headers/ArtificialNetwork.hpp"
#include "../Headers/RecurrentNetwork.hpp"

int main(int argc, char* argv[]) {

	// stbi_load();

	RecurrentNetwork network({3, 3}, 10);

	network.batch(
		// Inputs
		{{0, 0, 0},{0, 0, 1},{0, 1, 0},{0, 1, 1},
		 {1, 0, 0},{1, 0, 1},{1, 1, 0},{1, 1, 1}}, 

		// Corresponding expected outputs
		{{1, 1, 1},{1, 1, 0},{1, 0, 1},{1, 0, 0},
		 {0, 1, 1},{0, 1, 0},{0, 0, 1},{0, 0, 0}},

		// Total training cycles across threads
		500000,

		// Thread count
		16
	);

	//Network::printVector(network.perform({1, 0, 1}));

	network.perform({1, 0, 1});
	network.print();


	std::cout << "Cost: " << std::endl;
	Network::printVector(Network::calculateCost(network.perform({1, 0, 1}), {0, 1, 0}));

	network.save("savefile.noupload");

	return 0;
}
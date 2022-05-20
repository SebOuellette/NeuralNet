#include <iostream>
#include <cmath>
#include <iterator>
//#include "../Headers/NeuralNetwork.hpp"
//#include "../Headers/DeepNetwork.hpp"
#include "../Headers/Network.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../Libraries/stb/stb_image.h"

// Batch size of 1 for less than like infinite neurons, it's not faster
#define BATCH_SIZE 32
#define BATCH_SAMPLES 500

int main(int argc, char* argv[]) {

	// stbi_load();


	Network network({3, 10, 3});

	network.prepareBatches(BATCH_SIZE, BATCH_SAMPLES);

	network.batch(
		// Inputs
		{{0, 0, 0},{0, 0, 1},{0, 1, 0},{0, 1, 1},
		 {1, 0, 0},{1, 0, 1},{1, 1, 0},{1, 1, 1}}, 

		// Corresponding expected outputs
		{{1, 1, 1},{1, 1, 0},{1, 0, 1},{1, 0, 0},
		 {0, 1, 1},{0, 1, 0},{0, 0, 1},{0, 0, 0}}
	);

	//Network::printVector(network.perform({1, 0, 1}));

	network.perform({1, 0, 1});
	network.print();


	std::cout << "Cost: " << std::endl;
	Network::printVector(Network::calculateCost(network.perform({1, 0, 1}), {0, 1, 0}));


	//network.save("savefile.noupload");

	// Init
	// float x[INSTANCE_COUNT] = {0};
	// float y[INSTANCE_COUNT] = {0};
	// DeepNetwork network(4, 4, 2, INSTANCE_COUNT);

	// // Training
	// for (int bID=0;bID<BATCHES;bID++) {
		
	// 	for (int b=0;b<BATCH_CYCLES;b++) {
			
	// 		std::cout << "Danger Start..." << network.getInstances()->size() << std::endl;
	// 		for (int p=0;p<INSTANCE_COUNT;p++) {
	// 			std::cout << "Danger Looms overhead..." << std::endl;
	// 			Vector adjustment = network.prompt(p, {float(x[p]), float(y[p]), 5., 5.});

	// 			x[p] += (adjustment[0] * 2. - 1) * ADJUSTMENT_MULTIPLIER;
	// 			y[p] += (adjustment[1] * 2. - 1) * ADJUSTMENT_MULTIPLIER;
	// 		}
	// 		std::cout << "... Danger End" << std::endl;
	// 	}
		

	// 	// Kill the ones that didn't make it close enough
	// 	auto distanceToCenter = [](float x1, float y1){return sqrtf(powf(x1 - 5., 2.) + powf(y1 - 5., 2.));};
	// 	for (int p=0;p<INSTANCE_COUNT;p++) {
	// 		if (distanceToCenter(x[p], y[p]) > 5)
	// 			network.kill(p);
	// 	}
	// 	network.propagate();
	// }

	// // Display
	// for (int yd=0;yd<20;yd++) {
	// 	for (int xd=0;xd<20;xd++) {
	// 		bool drawInstance = false;

	// 		for (int i=0;i<INSTANCE_COUNT;i++) {
	// 			if ((int)round(x[i]) == xd && (int)round(y[i]) == yd)
	// 				drawInstance = true;
	// 		}

	// 		if (drawInstance)
	// 			std::cout << "# ";
	// 		else
	// 		 	std::cout << ". ";
	// 	}
	// 	std::cout << std::endl;
	// }

	return 0;
}
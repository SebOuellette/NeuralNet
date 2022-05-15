#include <iostream>
#include "../Headers/Network.hpp"

#define TRAINING_SAMPLES 300

int main(int argc, char* argv[]) {
	Network network(2, 3, 2);

	network.prompt({1, 2});

	return 0;
}
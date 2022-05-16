#include "../Headers/DeepNetwork.hpp"

Instance::Instance(int inputCount, int floatingCount, int outputCount) :
	Network(inputCount, floatingCount, outputCount) {}



DeepNetwork::DeepNetwork(int inputCount, int floatingCount, int outputCount, int instanceCount) :
	inputCount(inputCount),
	floatingCount(floatingCount),
	outputCount(outputCount),
	instanceCount(instanceCount) {
		
		for (int i=0;i<instanceCount;i++) {
			
		}
}
#include <math.h>
#include <stdexcept>
#include "Layer.hpp"

// This can have some odd results, but at lower testing counts,
// it can increase accuracy with a larget multiplier (like 10000). Play around with it to get optimal results
#define NEURON_CHANGE_MULTIPLIER 300


// Backpropagation Notes
/*
1. Change the bias. Move the bias an amount which is proportional to the 
"cost" of the output node.

2. Change the weights. If we want the output HIGH, then increase the 
weights between the output neuron, and the previous layer's neurons 
which are already high. Change must be proportional to the neuron's 
value (since they are multiplied). So if you have a high "cost" 
(difference),  change the weight a lot. If you have a low "cost", change 
it less.

3. Change the previous layer's neurons. Change the neuron's value 
proportionally to the weight connecting that neuron and the current 
neuron. We can't directly change it, so we save the desired change, 
and loop through all the output neurons, adding up all their desired 
changes. Once all the desired changes are collected, add the changes 
together to get a desired change for each node. Then go through the whole 
process again where the previous layer is the new "actual", and the 
"expected" is the previous layer + all the desired changes.
*/

typedef std::initializer_list<layer_s> SizeList;

class Network {
private:
	std::vector<Layer> layers;

public:
	// Constructor
	Network(layer_s, SizeList, layer_s);

	// Methods
	Vector askNetwork(Vector);
	Vector train(Vector, Vector);

	// Backprop methods
	Vector getPreviousWeights(index, index);
	float getPreviousBias(index, index);
	void backPropagate(Vector, Vector, index);

	void display();

	// Static methods
	static float calculateNetworkCost(Vector, Vector);
	static void PrintNeurons(Vector);
	static void PrintMatrix(Matrix);
};
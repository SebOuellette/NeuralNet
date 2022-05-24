#include "../Headers/RecurrentNetwork.hpp"

RecurrentNetwork::RecurrentNetwork(std::vector<int> neuronCounts, int lookbackSize) :
	Network(neuronCounts),
	lookbackSize(lookbackSize) {
	if (neuronCounts.size() < 2 || neuronCounts.size() > 3) {
		std::cerr << "Recurrent networks may only have 2 or 3 layers" << std::endl;
		exit(1);
	}

	this->previousOutputs.resize(lookbackSize);

	this->randomizeNetwork(neuronCounts);
}

RecurrentNetwork::RecurrentNetwork(std::vector<int> neuronCounts, std::string filename, int lookbackSize) :
	Network(neuronCounts, filename),
	lookbackSize(lookbackSize) {
	if (neuronCounts.size() < 2 || neuronCounts.size() > 3) {
		std::cerr << "Recurrent networks may only have 2 or 3 layers" << std::endl;
		exit(1);
	}

	this->previousOutputs.resize(lookbackSize);

	this->loadNetwork(neuronCounts, filename);
}

std::vector<int> RecurrentNetwork::getWeightSize(int layer) {
	if (layer >= this->values.size() - 1) {
		std::cerr << "Layer " << layer << " out of bounds for weights array" << std::endl;
		exit(1);
	}

	// Recurrent networks depend on all neurons for input
	// so we need to add up all the neurons
	int totalInfluencingValues = 0;
	for (int i=0;i<layerCount;i++) {
		totalInfluencingValues += this->values[i].size();
	}

	// Assumes worst case scenario, all neurons are populated
	for (int i=0;i<lookbackSize;i++) {
		totalInfluencingValues += this->values.back().size();
	}

	return {
		// Height / Rows
		int(this->values[layer + 1].size()),

		// Width / Columns
		totalInfluencingValues
	};
}

void RecurrentNetwork::performBackend(Vector input, Network* thisCopy) {};

void RecurrentNetwork::performBackend(Vector input, RecurrentNetwork* thisCopy) {
	// Add the old output to the previousOutputs array
	thisCopy->previousOutputs.insert(thisCopy->previousOutputs.begin(), thisCopy->values.back());

	// If the list is at it's max size, remove the oldest output vector from the list
	if (thisCopy->previousOutputs.size() > thisCopy->lookbackSize)
		thisCopy->previousOutputs.pop_back();

	thisCopy->setValues(0, input); // Just to make it easy
	
	int layerCount = values.size();

	// Generate the input for calculating the layer, aka compiling all
	// neurons in the network into one input vector
	Vector inputNeurons;

	// Append all the neurons in the current network
	for (int i=0;i<thisCopy->layerCount;i++) {
		for (float value : thisCopy->values[i]) {
			inputNeurons.push_back(value);
		}
	}

	// Append previous n outputs
	for (int i=0;i<thisCopy->lookbackSize;i++) {
		for (float value : thisCopy->previousOutputs[i]) {
			inputNeurons.push_back(value);
		}
	}
	// Propagate through the network
	for (int l=0;l<layerCount - 1;l++) {
		thisCopy->setValues(l+1, calculateLayer(inputNeurons, thisCopy->getWeights()[l], thisCopy->getBiases()[l]));
	}
}

void RecurrentNetwork::batch(Matrix input, Matrix expectedOutput, int trainingCycles, int batchSize) {
	for (Vector expected : expectedOutput) {
		if (this->values.back().size() != expected.size()) {
			std::cerr << "Network batch: expected output not the same size as actual output" << std::endl;
			exit(1);
		}
	}

	if (input.size() != expectedOutput.size()) {
		std::cerr << "Network batch: Must provide the same number of inputs and corresponding outputs" << std::endl;
		exit(1);
	}

	auto averageInto = [&](RecurrentNetwork* actualNetwork, RecurrentNetwork* secondNetwork){
		// Average out the current training data and the result of the training data in this copy
		// Values
		for (int v=0;v<actualNetwork->values.size();v++) {
			for (int r=0;r<actualNetwork->values[v].size();r++) {
				actualNetwork->values[v][r] += secondNetwork->values[v][r];
				actualNetwork->values[v][r] /= 2.;
			}
		}

		// Biases
		for (int b=0;b<actualNetwork->biases.size();b++) {
			for (int r=0;r<actualNetwork->biases[b].size();r++) {
				actualNetwork->biases[b][r] += secondNetwork->biases[b][r];
				actualNetwork->biases[b][r] /= 2.;
			}
		}

		// Weights
		for (int w=0;w<actualNetwork->weights.size();w++) {
			std::vector<int> weightSize = actualNetwork->getWeightSize(w);

			for (int r=0;r<weightSize[0];r++) {
				for (int c=0;c<weightSize[1];c++) {
					actualNetwork->weights[w][r][c] += secondNetwork->weights[w][r][c];
					actualNetwork->weights[w][r][c] /= 2.;
				}
			}
		}

		// Previous Outputs
		for (int o=0;o<actualNetwork->previousOutputs.size();o++) {
			for (int r=0;r<actualNetwork->previousOutputs[o].size();r++) {
				actualNetwork->previousOutputs[o][r] += secondNetwork->previousOutputs[o][r];
				actualNetwork->previousOutputs[o][r] /= 2.;
			}
		}
	};

	std::vector<std::thread*> threads(batchSize);
	std::vector<RecurrentNetwork*> networkCopies(batchSize);

	for(int t=0;t<batchSize;t++) {
		auto threadFunc = [&](RecurrentNetwork* thisCopy, RecurrentNetwork* actualNetwork) {

			for (int trainingCycle=0;trainingCycle<trainingCycles / batchSize;trainingCycle++) {
				
				// Cycle through the training data to ensure we evenly spread the training
				thisCopy->performBackend(input[trainingCycle % input.size()], thisCopy);
				Vector expected = expectedOutput[trainingCycle % expectedOutput.size()];

				std::vector<float*> lastValues;

				// Append all the neurons in the current network
				for (int i=0;i<thisCopy->layerCount;i++) {
					for (float value : thisCopy->values[i]) {
						lastValues.push_back(&value);
					}
				}

				// Append previous n outputs
				for (int i=0;i<thisCopy->lookbackSize;i++) {
					for (float value : thisCopy->previousOutputs[i]) {
						lastValues.push_back(&value);
					}
				}

				// Loop through all layers (recursion but without the downsides)
				int layer = thisCopy->layerCount;
				while (layer-- > 1) {
					
					Vector* actualLastValues = &thisCopy->values[layer - 1];
					Vector* biases = &thisCopy->biases[layer - 1];
					Matrix* weights = &thisCopy->weights[layer - 1];

					// For Step 3
					Vector neuronAdjustments(thisCopy->values[layer - 1].size());

					Vector cost = calculateCost(thisCopy->values[layer], expected);

					// Train the connections to each output neuron
					for (int r=0;r<thisCopy->values[layer].size();r++) {
						
						// Step 1: Adjust Bias
						(*biases)[r] += cost[r] / BIAS_ADJUST_DIVISOR;
						
						// Step 2: Adjust weights
						for (int l=0;l<lastValues.size();l++) {
							(*weights)[r][l] += cost[r] * *lastValues[l] / WEIGHT_ADJUST_DIVISOR;
						}

						// Step 3: Adjust neurons
						for (int l=0;l<actualLastValues->size();l++) {
							neuronAdjustments[l] += cost[r] * (*actualLastValues)[l] / NEURON_ADJUST_DIVISOR;
						}
					}

					// // Continuing Step 3: Create the new expected value for the layer before
					expected.resize(actualLastValues->size());
					for (int i=0;i<expected.size();i++) {
						expected[i] = (*actualLastValues)[i] + neuronAdjustments[i];
					}
				}
			}

			// Only if we could get write access, add to the main thread
			// If somebody si already writing, we just skip and add to our local thread again. 
			// This solution has no waiting times, and no lost computations.
			if (actualNetwork->lock()) {
				averageInto(actualNetwork, thisCopy);
				*thisCopy = *actualNetwork;

				actualNetwork->unlock();
			}
			
		};

		//A copy of this is made for each thread, afterwards they are all joined together
		networkCopies[t] = new RecurrentNetwork(*this);
		RecurrentNetwork* thisCopy = networkCopies[t];

		// thisCopy must be passed as an argument or it corrupts and you get segfaults
		threads[t] = new std::thread(threadFunc, thisCopy, this);
	}

	// Join the threads
	for (int i=0;i<threads.size();i++) {
		threads[i]->join();
		delete threads[i];
	}

	// Join the copies into the actual network for a final time, just incase any were missed
	for (int i=0;i<networkCopies.size();i++) {
		averageInto(this, networkCopies[i]);

		delete networkCopies[i];
	}
}
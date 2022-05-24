#include "../Headers/ArtificialNetwork.hpp"

ArtificialNetwork::ArtificialNetwork(std::vector<int> neuronCounts) :
	Network(neuronCounts) {
	if (neuronCounts.size() != 2 && neuronCounts.size() != 3) {
		std::cerr << "Artificial Network must have 2 or 3 layers given in constructor" << std::endl;
		exit(1);
	}
}

ArtificialNetwork::ArtificialNetwork(std::vector<int> neuronCounts, std::string filename) : 
	Network(neuronCounts, filename) {
	if (neuronCounts.size() != 2 && neuronCounts.size() != 3) {
		std::cerr << "Artificial Network must have 2 or 3 layers given in constructor" << std::endl;
		exit(1);
	}
}

std::vector<int> ArtificialNetwork::getWeightSize(int layer) {
	if (layer >= this->values.size() - 1) {
		std::cerr << "Layer " << layer << " out of bounds for weights array" << std::endl;
		exit(1);
	}

	return {
		// Height / Rows
		int(this->values[layer + 1].size()),

		// Width / Columns
		int(this->values[layer].size())
	};
}

// Multiply two matrices together
// Works even if one or more is a vector (still has to be a 2d vector, 
// but there is just one element in the second dimension)
Vector ArtificialNetwork::calculateLayer(Vector vector, Matrix matrix, Vector biases) {
	if (vector.size() != matrix[0].size()) {
		std::cerr << "calculateLayer-Network.cpp: Vector and matrix are incompatible, size mismatch."
		 << "Vector: " << vector.size() << " Matrix: " << matrix[0].size() << std::endl;
		exit(1);
	}

	Vector result;
	// The size of matrix is the number of rows, which is also
	// the number of rows of the output vector
	for (int r=0;r< matrix.size() ;r++) {
		float sum = 0;

		// Loop through all the "input" neurons and multiply by the weights matrix columns
		for (int i=0;i< vector.size() ;i++) {
			sum += vector[i] * matrix[r][i];
		}

		// When multiplying matrices, all the columns are multiplied by the
		// rows of the input vector, then those products are added together to get
		// the row of the output vector. so now we need to put that sum in the 
		// next row of the result

		// Add the bias corrosponding to this neuron
		sum += biases[r];

		// Finally, the resulting value must be constrained
		sum = ReLU(sum);

		// Next we set the "sum" as the row of the vector
		result.push_back(sum);
	}

	return result;
}

// Propagate through the network
void ArtificialNetwork::performBackend(Vector input, Network* thisCopy) {
	thisCopy->setValues(0, input); // Just to make it easy

	int layerCount = thisCopy->getValues().size();

	// Propagate through the network
	for (int l=0;l<layerCount - 1;l++) {
		thisCopy->setValues(l+1, calculateLayer(thisCopy->getValues()[l], thisCopy->getWeights()[l], thisCopy->getBiases()[l]));
	}
}


void ArtificialNetwork::batch(Matrix input, Matrix expectedOutput, int trainingCycles, int batchSize) {
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

	auto averageInto = [&](ArtificialNetwork* actualNetwork, ArtificialNetwork* secondNetwork){
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
	};

	std::vector<std::thread*> threads(batchSize);
	std::vector<ArtificialNetwork*> networkCopies(batchSize);

	for(int t=0;t<batchSize;t++) {
		auto threadFunc = [&](ArtificialNetwork* thisCopy, ArtificialNetwork* actualNetwork) {

			for (int trainingCycle=0;trainingCycle<trainingCycles / batchSize;trainingCycle++) {
				
				// Cycle through the training data to ensure we evenly spread the training
				thisCopy->performBackend(input[trainingCycle % input.size()], thisCopy);
				Vector expected = expectedOutput[trainingCycle % expectedOutput.size()];

				// Loop through all layers (recursion but without the downsides)
				int layer = thisCopy->layerCount;
				while (layer-- > 1) {

					Vector* lastValues = &thisCopy->values[layer - 1];
					Vector* biases = &thisCopy->biases[layer - 1];
					Matrix* weights = &thisCopy->weights[layer - 1];

					// For Step 3
					Vector neuronAdjustments(lastValues->size());

					Vector cost = calculateCost(thisCopy->values[layer], expected);

					// Train the connections to each output neuron
					for (int r=0;r<thisCopy->values[layer].size();r++) {
						
						// Step 1: Adjust Bias
						(*biases)[r] += cost[r] / BIAS_ADJUST_DIVISOR;
						
						// Step 2: Adjust weights
						for (int l=0;l<lastValues->size();l++) {
							(*weights)[r][l] += cost[r] * (*lastValues)[l] / WEIGHT_ADJUST_DIVISOR;
						}

						// Step 3: Adjust neurons
						for (int l=0;l<lastValues->size();l++) {
							neuronAdjustments[l] += cost[r] * (*lastValues)[l] / NEURON_ADJUST_DIVISOR;
						}
					}

					// // Continuing Step 3: Create the new expected value for the layer before
					expected.resize(lastValues->size());
					for (int i=0;i<expected.size();i++) {
						expected[i] = (*lastValues)[i] + neuronAdjustments[i];
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
		networkCopies[t] = new ArtificialNetwork(*this);
		ArtificialNetwork* thisCopy = networkCopies[t];

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
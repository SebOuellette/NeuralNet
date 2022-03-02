#include <initializer_list>
#include <vector>
#include <random>
#include <time.h>
#include "Functions.hpp"

#define INITIAL_WEIGHT_RANGE 1
#define INITIAL_WEIGHT_ACCURACY 1000

// Stored as [Y][X] or [Row][Column]
typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;
// Number of neurons in a layer
typedef unsigned int layer_s;
// Type of a layer. Input, Hidden, or Output
typedef enum {Input, Hidden, Output} LayerType;

// Layer class
class Layer {
private:
	Vector neurons;
	Matrix weights;
	Vector biases;
	const LayerType type;

	// Private Methods
	// Initialize to random values
	void InitializeWeights(layer_s, layer_s);
	void InitializeBiases(layer_s);
	// Initialize to 0
	void InitializeNeurons(layer_s);

public:
	// Constructors
	Layer(layer_s, layer_s, LayerType);
	Layer(layer_s, LayerType);

	// Getters
	Matrix getWeights();
	Vector getNeurons();
	Vector getBias();
	LayerType getType();

	// Setters
	void setNeurons(Vector);

	// Methods
	void setNextNeurons(Layer&);
};
#include <initializer_list>
#include <vector>
#include <random>
#include <time.h>
#include "Functions.hpp"

// Stored as [Y][X] or [Row][Column]
typedef std::vector<float> Vector;
typedef std::vector<Vector> Matrix;
// Number of neurons in a layer
typedef unsigned int layer_s;
// Index to be used when accessing vectors and matrices
typedef unsigned int index;
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
	Vector getBiases();
	LayerType getType();

	// Setters
	void setNeurons(Vector);
	void moveBias(index, float);
	void moveWeight(index, index, float);

	// Methods
	void setNextNeurons(Layer&);
};
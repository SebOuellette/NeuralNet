#include "../Headers/Functions.hpp"
#include <cmath>

// Logistic curve
float ReLU(float x) {
	return 1 / (1 + pow(e, -x));
}

// Squared difference of two values
// This is an adjusted cost function from what's normally used.
// This just makes more sense in implementation
float localCost(float expected, float actual) {
	return powf(expected - actual, 3);
}
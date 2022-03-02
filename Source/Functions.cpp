#include "../Headers/Functions.hpp"
#include <cmath>

// Logistic curve
float ReLU(float x) {
	return 1 / (1 + pow(e, -x));
}

// Squared difference of two values
float localCost(float expected, float actual) {
	return powf(expected - actual, 2.f);
}
#include "../Headers/Functions.hpp"
#include <cmath>

// Logistic curve
float ReLU(float x) {
	return 1.f / (1.f + powf(e, -x));
}

// Squared difference of two values
// This is an adjusted cost function from what's normally used.
// This just makes more sense in implementation
float localCost(float expected, float actual) {
	float cost = powf(actual - expected, 2.f);

	if (actual > expected)
		cost *= -1;
	return cost;
}
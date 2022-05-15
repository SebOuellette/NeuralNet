#include "../Headers/Functions.hpp"
#include <cmath>

// Logistic curve
float ReLU(float x) {
	return 1.f / (1.f + powf(e, -x));
}

float weightBiasClamp(float x) {
	return 2.f / (1.f + powf(e, -x)) - 1.f;
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

// Generates number between -4 and 4
//		Works great for weight generation (generally take the form of -4 - 4)
// 		Works great for neurons (ReLU function smushes -1 to -4 and 1 to 4 anyway)
float getRandom(float low, float high) {
	return std::fmod((float)rand() / 1000000, high - low) + low;
}
#include "../Headers/Functions.hpp"
#include <cmath>

// Logistic curve // Smooth 0. - 1.
float ReLU(float x) {
	return 1.f / (1.f + powf(e, -x));
}

// Generate a random float between -4 and 4
//		-4 and 4 because these values are used for weights
// 		Arguments accepted to change the default values
float getRandom(float low, float high) {
	return std::fmod((float)rand() / 1000000, high - low) + low;
}
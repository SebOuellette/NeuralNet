#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <math.h>

#define e 2.71828182846

float ReLU(float x);

float weightBiasClamp(float x);

float localCost(float expected, float actual);

float getRandom(float low = -4, float high = 4);

#endif // FUNCTIONS_HPP
#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <math.h>

#define e 2.71828182846

float ReLU(float x);

float weightBiasClamp(float x);

float localCost(float expected, float actual);

#endif // FUNCTIONS_HPP
#ifndef RECURRENT_NETWORK_HPP
#define RECURRENT_NETWORK_HPP



#include "Network.hpp"

class RecurrentNetwork : public Network {
private:
public:
	RecurrentNetwork(std::vector<int> neuronCounts);
	RecurrentNetwork(std::vector<int> neuronCounts, std::string filename);
};

#endif // RECURRENT_NETWORK_HPP
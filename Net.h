#pragma once
#include "Digits.h"

double dot(vector<double> v, vector<double> w) {
	assert(v.size() == w.size());
	double d = 0;
	for (size_t i = 0; i < v.size(); i++) {
		d += (v[i] * w[i]);
	}
	return d;
}

enum ACTIVATION_FUNCTION {
	SIGMOID,
	HEAVISIDE,
	RELU
};

double sigmoid(double x) {
	return 1 / (1 + exp(-1 * x));
}

double sigmoidDerivative(double x) {
	return sigmoid(x) * (1.0 - sigmoid(x));
}

double heaviside(double x) {
	if (x <= 0) {
		return 0;
	}
	return 1;
}

double heavisideDerivitive() {
	return 0;
}

double relu(double x) {
	if (x > 0) {
		return x;
	}
	return 0;
}

vector<double> softmax(vector<double> input) {
	vector<double> output;
	double norm = 0.0f;
	for (size_t i = 0; i < input.size(); i++) {
		double ex = exp(input[i]);
		output.push_back(ex);
		norm += ex;
	}
	for (size_t i = 0; i < output.size(); i++) {
		output[i] = output[i] / norm;
	}
	return output;
}

class Node {
public:
	double value;
	Node() {
		value = 0.0f;
	}
};

class Weight {
public:
	double value;
	Weight() {
		value = RAND_UNI;
		if (rand() % 2) {
			value *= -1;
		}
	}
};

class Net {
	vector<vector<Node>> nodes;
	vector<vector<vector<Weight>>> weights;
	double biasVal = NULL;
	double learningRate = 1;
public:
	Net() {};
	Net(size_t numInputs, size_t numOutputs, vector<size_t> hiddenLayers, double lr, double bias = NULL) {
		//hasBias = useBias;
		biasVal = bias;
		learningRate = lr;
		vector<Node> inputLayer = vector<Node>();
		for (size_t i = 0; i < numInputs; i++) {
			inputLayer.push_back(Node());
		}
		nodes.push_back(inputLayer);
		inputLayer.clear();

		for (size_t i = 0; i < hiddenLayers.size(); i++) {
			vector<Node> newLayer = vector<Node>();
			if (hiddenLayers[i] <= 0) {
				throw std::invalid_argument("size of hidden layer cannot be less than 1");
			}
			for (size_t j = 0; j < hiddenLayers[i]; j++) {
				newLayer.push_back(Node());
			}
			nodes.push_back(newLayer);
		}

		vector<Node> outputLayer = vector<Node>();
		for (size_t i = 0; i < numOutputs; i++) {
			outputLayer.push_back(Node());
		}
		nodes.push_back(outputLayer);

		for (size_t i = 0; i < nodes.size() - 1; i++) {
			weights.push_back(vector<vector<Weight>>());
			for (size_t j = 0; j < nodes[i].size(); j++) {
				//nodes[i][j] = working node
				weights[i].push_back(vector<Weight>());
				for (size_t k = 0; k < nodes[i + 1].size(); k++) {
					//nodes[i+1][k] = forward-connecting node
					weights[i][j].push_back(Weight());
				}
			}
		}

		if (bias != NULL) {
			Node b = Node();
			b.value = bias;
			for (size_t i = 0; i < nodes.size() - 1; i++) {
				nodes[i].push_back(b);
			}
			for (size_t i = 0; i < nodes.size() - 1; i++) {
				vector<Weight> biasWeights;
				size_t m = 0;
				if (i != nodes.size() - 2) {
					m = 1;
				}
				for (size_t j = 0; j < nodes[i + 1].size() - m; j++) {
					biasWeights.push_back(Weight());
				}
				weights[i].push_back(biasWeights);
			}
		}
	}
	void randomizeWeights() {
		for (size_t i = 0; i < weights.size(); i++) {
			for (size_t j = 0; j < weights[i].size(); j++) {
				for (size_t k = 0; k < weights[i][j].size(); k++) {
					weights[i][j][k] = Weight();
				}
			}
		}
	}
	Net(string filename) {
		this->readFile(filename);
	}
	void writeFile(string filename) {
		typedef numeric_limits<double> dbl;
		ofstream outFile;
		outFile.open(filename);
		if (!outFile.is_open()) {
			cout << "Could not write file: " << filename << '\n';
			return;
		}
		outFile.precision(dbl::max_digits10);
		if (biasVal == NULL) {
			outFile << "NULL" << '\n';
		}
		else {
			outFile << biasVal << fixed << '\n';
		}
		outFile << nodes.size() << '\n';
		for (size_t i = 0; i < nodes.size(); i++) {
			if (biasVal != NULL && i != nodes.size() - 1) {
				outFile << nodes[i].size() - 1 << '\n';
			}
			else {
				outFile << nodes[i].size() << '\n';
			}
		}
		for (size_t i = 0; i < weights.size(); i++) {
			for (size_t j = 0; j < weights[i].size(); j++) {
				for (size_t k = 0; k < weights[i][j].size(); k++) {
					outFile << weights[i][j][k].value << fixed << '\n';
				}
			}
		}
		outFile.close();
	}
	void readFile(string filename) {
		ifstream inFile;
		inFile.open(filename);
		if (!inFile.is_open()) {
			cout << "Could not open file: " << filename << '\n';
		}
		double biasVal;
		size_t inputSize;
		size_t outputSize;
		vector<size_t> hiddenLayers;
		size_t totalLayers;
		string line;
		getline(inFile, line);
		if (line == "NULL") {
			biasVal = NULL;
		}
		else {
			biasVal = stod(line);
		}
		getline(inFile, line);
		totalLayers = stoi(line);

		getline(inFile, line);
		inputSize = stoul(line);

		for (size_t i = 1; i < totalLayers - 1; i++) {
			getline(inFile, line);
			size_t n = stoul(line);
			hiddenLayers.push_back(n);
		}

		getline(inFile, line);
		outputSize = stoul(line);

		Net rNet(inputSize, outputSize, hiddenLayers, biasVal);

		for (size_t i = 0; i < rNet.weights.size(); i++) {
			for (size_t j = 0; j < rNet.weights[i].size(); j++) {
				for (size_t k = 0; k < rNet.weights[i][j].size(); k++) {
					getline(inFile, line);
					rNet.weights[i][j][k].value = stod(line);
				}
			}
		}

		*this = rNet;
	}
	void test(vector<double> input) {
		assert((nodes.size() != 0) && (nodes.size() != 1) && (nodes[0].size() != 0));
		if (biasVal != NULL) {
			assert(nodes[0].size() - 1 == input.size());
		}
		else {
			assert(nodes[0].size() == input.size());
		}
		for (size_t i = 0; i < input.size(); i++) {
			nodes[0][i].value = input[i];
		}

		for (size_t i = 1; i < nodes.size(); i++) {
			size_t m = 0;
			if (biasVal != NULL && i != nodes.size() - 1) {
				m = 1;
			}
			for (size_t j = 0; j < nodes[i].size() - m; j++) {
				vector<double> v;
				vector<double> w;
				for (size_t k = 0; k < nodes[i - 1].size(); k++) {
					v.push_back(nodes[i - 1][k].value);
					w.push_back(weights[i - 1][k][j].value);
				}
				double d = dot(v, w);
				if (i == nodes.size() - 1) {
					if (getOutputSize() == 1) {
						nodes[i][j].value = sigmoid(d);
					}
					else {
						nodes[i][j].value = d;
					}
				}
				else {
					nodes[i][j].value = sigmoid(d);
				}
			}
		}

		if (getOutputSize() > 1) {
			vector<double> currentOutputs = getOutputs();
			currentOutputs = softmax(currentOutputs);
			emplaceOutputs(currentOutputs);
		}
		
	}
	double meanSquaredError(vector<double> expectedOutput) {
		double mSE = 0.0f;
		vector<double> predictedOutputs = getOutputs();
		for (size_t i = 0; i < expectedOutput.size(); i++) {
			mSE += pow(expectedOutput[i] - predictedOutputs[i], 2);
		}
		mSE = mSE / (double)expectedOutput.size();
		return mSE;
	}
	void train(vector<double> input, vector<double> expectedOutput) {
		this->test(input);
		assert(nodes[nodes.size() - 1].size() == expectedOutput.size());

		vector<vector<vector<Weight>>> newWeights = weights;

		for (size_t i = 0; i < weights.size(); i++) {
			for (size_t j = 0; j < weights[i].size(); j++) {
				for (size_t k = 0; k < weights[i][j].size(); k++) {

					//cout << "Working Weight: " << i << ',' << j << ',' << k << '\n';

					vector<vector<size_t>> affectedWeights;
					//push back working weight
					affectedWeights.push_back({ i,j,k });

					for (size_t l = weights.size() - 1; l > i; l--) {
						//l,m,n = layer,group,individual
						if (l == i + 1) {
							//get weights in next layer of same individual
							for (size_t n = 0; n < weights[l][k].size(); n++) {
								//cout << "Ahead Weight: " << l << ',' << j << ',' << n << '\n';

								affectedWeights.push_back({ l,k,n });
							}
						}
						else {
							//get all further weights except bias
							size_t x = 0;
							if (biasVal != NULL) {
								x = -1;
							}
							for (size_t m = 0; m < weights[l].size() + x; m++) {
								for (size_t n = 0; n < weights[l][m].size(); n++) {
									//cout << "Ahead Weight: " << l << ',' << m << ',' << n << '\n';

									affectedWeights.push_back({ l,m,n });
								}
							}
						}
						
					}

					reverse(affectedWeights.begin(), affectedWeights.end());

					//cout << '\n';
					double delta = 1.0f;
					for (size_t w = 0; w < affectedWeights.size(); w++) {
						
						//iterate foreach affected weight, starting with outermost
						size_t l = affectedWeights[w][0];
						size_t m = affectedWeights[w][1];
						size_t n = affectedWeights[w][2];
						//l,m,n = layer,group,individual
						if (l == weights.size() - 1) {
							//if weight is connected directly to output
							double output = nodes[l + 1][n].value;
							double target = expectedOutput[n];
							double dEO = -1 * (target - output);
							double dON = output * (1 - output);

							delta *= dEO;
							delta *= dON;
						}
						if (w == affectedWeights.size() - 1) {
							//reached final weight
							double outputBehind = nodes[l][m].value;
							double dNW = outputBehind;

							delta *= dNW;
						}
						else {
							//not there yet
							double dNO = weights[l][m][n].value;
							delta *= dNO;
						}
					}
					//wrap it all up
					//at this posize_t 'delta' should represent the effect that this specific weight has on the
					//total error of the system
					double wplus = weights[i][j][k].value - (learningRate * delta);
					newWeights[i][j][k].value = wplus;
				}
			}
		}
		weights = newWeights;
	}
	vector<double> getOutputs() {
		vector<double> o;
		for (size_t i = 0; i < nodes[nodes.size() - 1].size(); i++) {
			o.push_back(nodes[nodes.size() - 1][i].value);
		}
		return o;
	}
	void emplaceOutputs(vector<double> newOutputs) {
		for (size_t i = 0; i < nodes[nodes.size() - 1].size(); i++) {
			nodes[nodes.size() - 1][i].value = newOutputs[i];
		}
	}
	vector<size_t> getHiddenSizes() {
		//returns size of each hidden layer
		//bias nodes are not counted
		vector<size_t> sizes;
		for (size_t i = 1; i < nodes.size() - 1; i++) {
			size_t s = nodes[i].size();
			if (biasVal != NULL) {
				s--;
			}
			sizes.push_back(s);
		}
		return sizes;
	}
	size_t getInputSize() {
		if (biasVal != NULL) {
			return nodes[0].size() - 1;
		}
		return nodes[0].size();
	}
	size_t getOutputSize() {
		return nodes[nodes.size() - 1].size();
	}
	double getLearningRate() {
		return learningRate;
	}
	void setLearningRate(double newLearningRate) {
		learningRate = newLearningRate;
	}
	double getBiasVal() {
		return biasVal;
	}
	size_t getMeanOfInputAndOutput() {
		size_t avg = size_t((getInputSize() + getOutputSize()) / 2);
		return avg;
	}
};
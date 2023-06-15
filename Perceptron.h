#pragma once
#include "Digits.h"

class Perceptron {
	vector<double> input;
	double output;
	vector<double> weights;
	double lr;
	double sigmoid(double x) {
		return 1 / (1 + exp(-1 * x));
	}
public:
	Perceptron(size_t inputSize, double learningRate) {
		//include 1 extra for bias
		input.resize(inputSize);
		input.push_back(1);
		output = 0.0f;
		lr = learningRate;
		for (size_t i = 0; i < input.size(); i++) {
			weights.push_back(RAND_UNI);
		}
	}
	void emplaceInput(vector<double> newInputs) {
		assert(newInputs.size() == input.size() - 1);
		for (size_t i = 0; i < newInputs.size(); i++) {
			input[i] = newInputs[i];
		}
	}
	void feedForward() {
		//for each output
		double sum = 0;
		for (size_t j = 0; j < input.size(); j++) {
			double value = input[j];
			double weight = weights[j];
			sum += value * weight;
		}
		output = sigmoid(sum);
	}
	double getOutput() {
		return output;
	}
	double calculateError(double expectedOutput) {
		double e = expectedOutput - getOutput();
		return e;
	}
	void updateWeights(double error) {
		for (size_t i = 0; i < weights.size(); i++) {
			weights[i] += error * input[i] * lr;
		}
	}
	void train(vector<double> input, double expectedOutput) {
		emplaceInput(input);
		feedForward();
		double error = calculateError(expectedOutput);
		updateWeights(error);
	}
	void test(vector<double> input, double expectedOutput) {
		emplaceInput(input);
		feedForward();
	}
	double grade(vector<vector<double>> allInputs, vector<double> allExpectedOutputs) {
		unsigned correct = 0;
		for (size_t i = 0; i < allInputs.size(); i++) {
			test(allInputs[i], allExpectedOutputs[i]);
			double output = getOutput();
			if (output == allExpectedOutputs[i]) {
				correct++;
			}
		}
		double score = double(correct) / double(allInputs.size());
		return score;
	}
};
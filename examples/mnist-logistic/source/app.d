import std.algorithm;
import std.array;
import std.range;
import std.stdio;

import dnnet;

import data;

void main(string[] args)
{
	if(args.length < 2)
	{
		writeln("Usage: ", args[0], " <MNIST path>");
		return;
	}

	//Define the hyperparameters up here
	const int epochs = 1000;
	const int batchSize = 500;
	const float learningRate = 0.1;

	//Load the MNIST features and labels
	auto data = loadMNIST(args[1]);
	auto trainFeatures = data.trainFeatures;
	auto trainLabels = data.trainLabels;
	auto testFeatures = data.testFeatures;
	auto testLabels = data.testLabels;

	//Declare the layers that we wish to use as inputs to the network.
	//In this case it is only the features and labels for the MNIST dataset.
	auto featuresLayer = datasource([batchSize, 28 * 28]);
	auto labelsLayer = datasource([batchSize, 10]);

	/*
	   Now we define the actual network architecture -- UFCS makes this pretty :)
	   Initialise weights and biases to values between -0.1 and 0.1
	*/
	auto layers = featuresLayer
				 .dense(10, uniformInit(-0.01, 0.01), uniformInit(-0.01, 0.01))
				 .softmax()
				 .crossentropy(labelsLayer);

	/*
		Create a Network object that can take care of compiling the network specific
		optimisation problem to CUDA code.
	
		The output layer of the model is specified by layers

		[featuresLayer, labelsLayer] indicates which order the model.trainBatch method
		should expect arguments to be passed in.

		sgd(...) specifies the optimisation algorithm we want to use to find the parameters.
	*/
	auto model = new Network(layers, [featuresLayer, labelsLayer], sgd(learningRate));

	//Train/evaluate the model!
	for(int e = 0; e < epochs; e++)
	{
		//Do an epoch of training
		foreach(features, labels; zip(trainFeatures.chunks(batchSize), trainLabels.chunks(batchSize)))
		{
			model.trainBatch([features.joiner().array(), labels.joiner().array()]);
		}

		float loss = 0.0f;
		int testBatches = 0;

		//Compute the average loss over the test batches
		foreach(features, labels; zip(testFeatures.chunks(batchSize), testLabels.chunks(batchSize)))
		{
			loss += model.evaluate([features.joiner().array(), labels.joiner().array()]);
			testBatches++;
		}

		//Write out the loss
		writeln("Epoch ", e + 1, " test error is ", loss / testBatches);
	}
}

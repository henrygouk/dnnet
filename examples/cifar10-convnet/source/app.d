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
		writeln("Usage: ", args[0], " <CIFAR10 path>");
		return;
	}

	//Define the hyperparameters up here
	const int epochs = 100;
	const int batchSize = 100;
	const float learningRate = 0.001;
	const float momentumRate = 0.9;

	//Load the CIFAR10 data
	auto data = loadCIFAR10(args[1]);
	auto trainFeatures = data.trainFeatures;
	auto trainLabels = data.trainLabels;
	auto testFeatures = data.testFeatures;
	auto testLabels = data.testLabels;

	//Create the input layers
	auto featuresLayer = datasource([batchSize, 3, 32, 32]);
	auto labelsLayer = datasource([batchSize, 10]);

	//Create the network
	auto layers = featuresLayer
				 .convolutional(128, [3, 3])
				 .relu()
				 .convolutional(128, [3, 3])
				 .maxpool([3, 3], [2, 2])
				 .relu()
				 .convolutional(192, [3, 3])
				 .relu()
				 .convolutional(192, [3, 3])
				 .maxpool([3, 3], [2, 2])
				 .relu()
				 .dense(128)
				 .dropout()
				 .relu()
				 .dense(128)
				 .dropout()
				 .relu()
				 .dense(10)
				 .softmax();

	auto lossLayer = layers.crossentropy(labelsLayer);

	auto model = new Network([lossLayer, layers], [featuresLayer, labelsLayer], adam(learningRate, momentumRate));

	//Train/evaluate the model!
	for(int e = 0; e < epochs; e++)
	{
		//Do an epoch of training
		foreach(features, labels; zip(trainFeatures.chunks(batchSize), trainLabels.chunks(batchSize)))
		{
			model.trainBatch([features.joiner().array(), labels.joiner().array()]);
		}

		float loss = 0.0f;
		float accuracy = 0.0f;
		int testBatches = 0;

		//Compute the average loss over the test batches
		foreach(features, labels; zip(testFeatures.chunks(batchSize), testLabels.chunks(batchSize)))
		{
			auto output = model.process([features.joiner().array(), labels.joiner().array()]);
			loss += output[0][0];

			foreach(pg; zip(output[1].chunks(10), labels))
			{
				accuracy += (pg[0].maxPos().length == pg[1].maxPos.length) ? (1.0 / batchSize) : 0.0;
			}

			testBatches++;
		}

		//Write out the loss
		writeln("Epoch ", e + 1, " test error is ", loss / testBatches, " and accuracy is ", accuracy / testBatches);
	}
}

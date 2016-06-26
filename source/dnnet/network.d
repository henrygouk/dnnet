/*
	The MIT License (MIT)

	Copyright (c) 2016 Henry Gouk

	Permission is hereby granted, free of charge, to any person obtaining a copy
	of this software and associated documentation files (the "Software"), to deal
	in the Software without restriction, including without limitation the rights
	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
	copies of the Software, and to permit persons to whom the Software is
	furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in
	all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
	THE SOFTWARE.
*/

module dnnet.network;

import std.algorithm;
import std.array;
import std.exception;

import dopt.core;

import dnnet;

class Network
{
	public
	{
		this(Layer outputLayer, Layer[] inputLayers, UpdateRule updateRule)
		{
			this([outputLayer], inputLayers, updateRule);
		}

		this(Layer[] outputLayers, Layer[] inputLayers, UpdateRule updateRule)
		{
			enforce(outputLayers.length > 0, "There must be at least one output layer.");
			enforce(outputLayers[0].expression.volume == 1 || updateRule is null, "The first output layer must produce a scalar.");

			//Check that the input layers are all variables
			foreach(input; inputLayers)
			{
				enforce(cast(Variable)input.expression !is null, "All input layers must have an expression that can be casted to a Variable.");
			}

			mOutputLayers = outputLayers.dup;
			mInputLayers = inputLayers.dup;

			auto inputs = inputLayers
						 .map!(x => cast(Variable)x.expression)
						 .array();

			void traverseNetwork(Layer l)
			{
				if(mLayers.canFind(l))
				{
					return;
				}

				foreach(d; l.deps)
				{
					traverseNetwork(d);
				}

				mLayers ~= l;
			}

			foreach(outputLayer; mOutputLayers)
			{
				traverseNetwork(outputLayer);
			}

			mParameters = mLayers
						 .map!(x => x.parameters)
						 .joiner()
						 .array();

			auto paramVars = mParameters
							.map!(x => x.variable)
							.array();

			auto networkFunction = func(inputs ~ paramVars, mOutputLayers.map!(x => x.expression).array());
			mKernel = compile!CUDACompiler(networkFunction);

			if(updateRule !is null)
			{
				mOptimiser = createOptimiser(updateRule, networkFunction, mParameters, inputs);
			}
		}

		void trainBatch(float[][] inputs)
		{
			enforce(mOptimiser !is null, "No update rule was provided during network creation.");

			mOptimiser(inputs);
		}

		float evaluate(float[][] inputs)
		{
			void[][] kernelInputs = inputs.map!(x => cast(void[])x).array() ~ mParameters.map!(x => cast(void[])x.value).array();
			return (cast(float[][])mKernel(kernelInputs))[0][0];
		}

		float[][] process(float[][] inputs)
		{
			import std.range;
			void[][] kernelInput = chain(inputs, parameters.map!(x => x.value))
								  .map!(x => cast(void[])x)
								  .array();

			void[][] kernelOutput = mKernel(kernelInput);

			return kernelOutput
				  .map!(x => cast(float[])x)
				  .array();
		}

		@property Parameter[] parameters()
		{
			return mParameters.dup;
		}
	}

	protected
	{
		Layer[] mOutputLayers;
		Layer[] mInputLayers;
		Layer[] mLayers;
		Parameter[] mParameters;
		Optimiser mOptimiser;
		CUDAKernel mKernel;
	}
}

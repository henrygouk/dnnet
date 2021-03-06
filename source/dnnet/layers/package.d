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

module dnnet.layers;

import std.algorithm;
import std.array;

import dopt.core;

import dnnet;

public
{
	import dnnet.layers.batchnorm;
	import dnnet.layers.convolutional;
	import dnnet.layers.crossentropy;
	import dnnet.layers.datasource;
	import dnnet.layers.deconvolutional;
	import dnnet.layers.dense;
	import dnnet.layers.dropout;
	import dnnet.layers.maxpool;
	import dnnet.layers.relu;
	import dnnet.layers.softmax;
	import dnnet.layers.squarederror;
	import dnnet.layers.tanh;
	import dnnet.layers.weightdecay;
}

class Layer
{
	public
	{
		this(Layer[] deps, Tensor expression, Parameter[] parameters)
		{
			mDeps = deps.dup;
			mExpression = expression;
			mTrainExpression = expression;
			mParameters = parameters.dup;
		}

		this(Layer[] deps, Tensor trainExpression, Tensor expression, Parameter[] parameters)
		{
			import std.exception;
			enforce(expression.shape == trainExpression.shape);

			mDeps = deps.dup;
			mExpression = expression;
			mTrainExpression = trainExpression;
			mParameters = parameters.dup;
		}

		@property Layer[] deps()
		{
			return mDeps.dup;
		}

		@property Tensor expression()
		{
			return mExpression;
		}

		@property Tensor trainExpression()
		{
			return mTrainExpression;
		}

		@property Parameter[] parameters()
		{
			return mParameters.dup;
		}

		@property const(uint)[] outputShape()
		{
			return mExpression.shape;
		}
	}

	protected
	{
		Layer[] mDeps;
		Tensor mExpression;
		Tensor mTrainExpression;
		Parameter[] mParameters;
	}
}

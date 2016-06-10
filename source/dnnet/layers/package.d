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
	import dnnet.layers.crossentropy;
	import dnnet.layers.datasource;
	import dnnet.layers.dense;
	import dnnet.layers.relu;
	import dnnet.layers.softmax;
}

class Layer
{
	public
	{
		this(Layer[] deps, Tensor expression, Parameter[] parameters)
		{
			mDeps = deps.dup;
			mExpression = expression;
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
		Parameter[] mParameters;
	}
}

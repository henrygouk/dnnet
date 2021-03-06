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

module dnnet.layers.weightdecay;

import std.algorithm;

import dopt.core;

import dnnet;

//Very naive method. Once the Parameter class has been improved this should change.
Parameter[] getRegularisedParams(Layer layer)
{
	//Get a list of all the layers
	Layer[] ls;

	void traverse(Layer l)
	{
		if(ls.canFind(l))
		{
			return;
		}

		ls ~= l;

		foreach(d; l.deps)
		{
			traverse(d);
		}
	}

	traverse(layer);

	//Now get the parameters that should be penalised
	Parameter[] weights;

	foreach(l; ls)
	{
		foreach(p; l.parameters)
		{
			if(p.regularisable)
			{
				weights ~= p;
			}
		}
	}

	return weights;
}

Layer weightdecay(Layer input, Parameter[] weights, float decayRate)
{
	auto sqWeights = weights
					.map!(x => sum(pow(x.expression, 2)) / x.expression.volume)
					.fold!"a + a";

	return new Layer([input], input.expression + sqWeights * (decayRate / weights.length), []);
}

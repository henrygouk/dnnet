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

module dnnet.layers.convolutional;

import std.algorithm;

import dopt.core;

import dnnet;

Layer convolutional(Layer input, uint numOutputs, in uint[] kernelShape)
{
	return convolutional(input, numOutputs, kernelShape, glorotInit(), constantInit(0.0f));
}

Layer convolutional(W, B)(Layer input, uint numOutputs, in uint[] kernelShape, W weightsInit, B biasesInit)
{
	auto w = Parameter.create(weightsInit, [numOutputs, input.outputShape[1]] ~ kernelShape);
	w.regularisable = true;
	auto b = Parameter.create(biasesInit, [numOutputs]);
	auto x = input.expression;

	auto z = new MultiFilter(x, w.expression, false, true, FilterMode.convolution);
	auto y = z + b.expression.repeat(z.shape[2 .. $].fold!"a * b").transpose().repeat(z.shape[0]).reshape(z.shape);

	return new Layer([input], y, [w, b]);
}

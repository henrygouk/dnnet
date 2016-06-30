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

module dnnet.layers.dense;

import dopt.core;

import dnnet;

Layer dense(W, B)(Layer input, uint numOutputs, W w = glorotInit(), B b = constantInit(0.0f))
{
	auto x = input.expression;
	x = x.reshape([x.shape[0], x.volume / x.shape[0]]);
	auto weights = Parameter.create(w, [x.shape[1], numOutputs]);
	weights.regularisable = true;
	auto biases = Parameter.create(b, [numOutputs]);

	return new Layer([input], new MatrixMultiply(x, weights.expression) + repeat(biases.expression, input.outputShape[0]), [weights, biases]);
}

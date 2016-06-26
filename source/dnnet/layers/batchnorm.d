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

module dnnet.layers.batchnorm;

import dopt.core;

import dnnet;

Layer batchnorm(Layer input)
{
	auto beta = new Parameter(float32());
	auto gamma = new Parameter(float32(), [1.0f]);
	auto x = input.expression;
	auto batchSize = x.shape[0];

	auto mean = (1.0f / batchSize) * repeat(sum(x, x.rank - 1), batchSize);
	auto var = (1.0f / batchSize) * repeat(sum(pow(x - mean, 2), x.rank - 1), batchSize);
	auto y = gamma.expression * (x - mean) / sqrt(var + 1e-4) + beta.expression;

	return new Layer([input], y, [beta, gamma]);
}

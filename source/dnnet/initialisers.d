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

module dnnet.initialisers;

import std.algorithm;
import std.exception;
import std.math;
import std.random;

auto constantInit(float val)
{
	float[] init(in uint[] shape)
	{
		auto params = new float[shape.fold!"a * b"(1)];
		params[] = val;

		return params;
	}

	return &init;
}

auto uniformInit(float low, float high)
{
	float[] init(in uint[] shape)
	{
		auto params = new float[shape.fold!"a * b"(1)];

		for(size_t i = 0; i < params.length; i++)
		{
			params[i] = uniform(low, high);
		}

		return params;
	}

	return &init;
}

auto glorotInit()
{
	float[] init(in uint[] shape)
	{
		enforce(shape.length > 0, "shape must have a nonzero length.");

		auto params = new float[shape.fold!"a * b"(1)];

		size_t weightVol = 1;
		
		if(shape.length > 2)
		{
			weightVol = shape[2 .. $].fold!"a * b"(1);
		}

		size_t fan;

		if(shape.length == 1)
		{
			fan = shape[0];
		}
		else
		{
			fan = shape[0] * shape[1];
		}

		float stdDev = sqrt(2.0f / (fan * weightVol));
		float r = stdDev * sqrt(3.0f);

		for(size_t i = 0; i < params.length; i++)
		{
			//params[i] = uniform(-r, r);
			import std.mathspecial;
			params[i] = stdDev * normalDistributionInverse(uniform(0.0, 1.0));
		}

		return params;
	}

	return &init;
}

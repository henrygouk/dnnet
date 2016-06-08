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

module dnnet.core;

import std.algorithm;
import std.array;

import dopt.core;

class Parameter
{
	public
	{
		this(Variable param)
		{
			mParameter = param;
			mValue = new float[param.volume];
			mValue[] = 0;
		}

		this(Variable param, float[] init)
		{
			mParameter = param;
			mValue = init.dup;
		}

		@property Variable variable()
		{
			return mParameter;
		}

		@property Tensor expression()
		{
			return mParameter;
		}

		@property float[] value()
		{
			return mValue;
		}

		static Parameter create(C)(ctor, in uint[] shape)
		{
			static if(is(C == Parameter))
			{
				enforce(ctor.variable.shape == shape, "The supplied parameter does not have the correct shape.");
				return ctor;
			}
			else static if(is(C == float[] delegate(in uint[])) || is(C == float[] function(in uint[])))
			{
				return new Parameter(float32(shape), ctor(shape));
			}
			else
			{
				static assert(0, "Unsupported parameter initialiser.");
			}
		}
	}

	protected
	{
		Variable mParameter;
		float[] mValue;
	}
}


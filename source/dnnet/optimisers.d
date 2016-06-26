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

module dnnet.optimisers;

import std.algorithm;
import std.array;
import std.exception;
import std.range;
import std.typecons;

import dopt.core;

import dnnet;

alias Update = Tuple!(Parameter, Tensor);
alias UpdateRule = Update[] delegate(Function, Update[]);
alias Optimiser = void delegate(float[][] inputs);

Update[] createUpdates(Parameter[] params)
{
	return zip(params, params.map!(x => cast(Tensor)x.variable)).array();
}

Variable[] getVariables(Update[] updates)
{
	return updates
		  .map!(x => x[0].variable)
		  .array();
}

UpdateRule sgd(float learningRate = 0.01f)
{
	auto updateRule(Function network, Update[] updates)
	{
		//updates = updates.dup;
		auto paramVars = getVariables(updates);
		auto grads = network.gradients(paramVars);

		for(size_t i = 0; i < updates.length; i++)
		{
			updates[i][1] = updates[i][0].variable - (grads[i] * learningRate);
		}

		return updates;
	}

	return &updateRule;
}

auto addMomentum(Update[] updates, float momentumRate) 
{
	updates = updates.dup;
	auto paramVars = getVariables(updates);
	
	//Create a new set of variables with the same structure as the params
	auto velocities = paramVars
					 .map!(x => new Variable(x.type))
					 .array();

	auto velocityUpdates = new Update[velocities.length];

	for(size_t i = 0; i < velocities.length; i++)
	{
		auto x = momentumRate * velocities[i] + updates[i][1];
		
		velocityUpdates[i][0] = new Parameter(velocities[i]);
		velocityUpdates[i][1] = x - updates[i][0].variable;
		updates[i][1] = x;
	}

	return updates ~ velocityUpdates;
}

UpdateRule momentum(float learningRate = 0.001f, float momentumRate = 0.9f)
{
	auto sgdRule = sgd(learningRate);

	auto updateRule(Function network, Update[] updates)
	{
		return addMomentum(sgdRule(network, updates), momentumRate);
	}

	return &updateRule;
}

UpdateRule rmsprop(float learningRate = 1.0f, float momentumRate = 0.9f, float eps = 1.0e-6)
{
	auto updateRule(Function network, Update[] updates)
	{
		updates = updates.dup;
		auto paramVars = getVariables(updates);
		auto grads = network.gradients(paramVars);

		auto velocityUpdates = new Update[paramVars.length];

		for(size_t i = 0; i < paramVars.length; i++)
		{
			velocityUpdates[i][0] = new Parameter(new Variable(paramVars[i].type));
			velocityUpdates[i][1] = momentumRate * velocityUpdates[i][0].variable + (1 - momentumRate) * pow(grads[i], 2);
			updates[i][1] = paramVars[i] - (learningRate * grads[i] / sqrt(velocityUpdates[i][1] + eps));
		}

		return updates ~ velocityUpdates;
	}

	return &updateRule;
}

UpdateRule adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1.0e-8)
{
	auto updateRule(Function network, Update[] updates)
	{
		updates = updates.dup;
		auto paramVars = getVariables(updates);
		auto grads = network.gradients(paramVars);

		auto b1 = float32([]);
		auto b2 = float32([]);
		auto nb1 = b1 * beta1;
		auto nb2 = b2 * beta2;
		auto alpha = learningRate * sqrt(1.0f - nb2) / (1.0f - nb1);

		Update[] newUpdates = [Update(new Parameter(b1, [1.0f]), nb1), Update(new Parameter(b2, [1.0f]), nb2)];

		for(size_t i = 0; i < paramVars.length; i++)
		{
			auto p = paramVars[i];
			auto m = new Variable(p.type);
			auto v = new Variable(p.type);
			auto g = grads[i];

			auto nm = beta1 * m + (1 - beta1) * g;
			auto nv = beta2 * v + (1 - beta2) * g * g;
			auto np = p - alpha * nm / (sqrt(nv) + eps);

			newUpdates ~= Update(new Parameter(m), nm);
			newUpdates ~= Update(new Parameter(v), nv);
			updates[i][1] = np;
		}

		return updates ~ newUpdates;
	}

	return &updateRule;
}

Optimiser createOptimiser(UpdateRule updateRule, Function network, Parameter[] parameters, Variable[] userParams)
{
	auto updates = updateRule(network, createUpdates(parameters));

	auto updateFunc = func(userParams ~ updates.map!(x => x[0].variable).array(), updates.map!(x => x[1]).array());
	auto kernel = compile!CUDACompiler(updateFunc);
	
	auto inputBuffers = updateFunc
					   .inputs
					   .map!(x => allocate!CUDACompiler(x.type))
					   .array();
	
	auto outputBuffers = inputBuffers[userParams.length .. $];

	void optimiser(float[][] inputs)
	{
		enforce(inputs.length == userParams.length, "Incorrect number of inputs supplied.");

		for(size_t i = 0; i < inputs.length; i++)
		{
			inputBuffers[i].set(inputs[i]);	
		}
		
		for(size_t i = 0; i < updates.length; i++)
		{
			outputBuffers[i].set(updates[i][0].value);
		}

		kernel.execute(inputBuffers, outputBuffers);

		for(size_t i = 0; i < updates.length; i++)
		{
			outputBuffers[i].get(updates[i][0].value);
		}
	}

	return &optimiser;
}

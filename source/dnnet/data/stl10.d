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

module dnnet.data.stl10;

import std.algorithm;
import std.array;
import std.conv;
import std.file;
import std.stdio;
import std.typecons;

auto loadSTL10(string directory)
{
	auto loadFeatures(string filename)
	{
		float[][] features;
		auto buf = new ubyte[3 * 96 * 96];

		auto fd = File(filename, "rb");

		while(fd.rawRead(buf).length == buf.length)
		{
			auto fs = new float[3 * 96 * 96];

			for(size_t c = 0; c < 3; c++)
			{
				for(size_t y = 0; y < 96; y++)
				{
					for(size_t x = 0; x < 96; x++)
					{
						fs[c * 96 * 96 + y * 96 + x] = (cast(float)buf[c * 96 * 96 + x * 96 + y] - 128.0) / 128.0;
					}
				}
			}

			features ~= fs;
		}

		return features;
	}

	auto loadLabels(string filename)
	{
		float[][] labels;
		
		auto ls = cast(ubyte[])read(filename);

		foreach(l; ls)
		{
			auto inst = new float[10];
			inst[] = 0.0;
			inst[l - 1] = 1.0;

			labels ~= inst;
		}

		return labels;
	}

	auto loadIndices(string filename)
	{
		size_t[][] folds;

		string line;
		auto fd = File(filename);

		while((line = fd.readln()) !is null)
		{
			folds ~= line.split()
					.map!(x => x.to!size_t)
					.array();
		}

		return folds;
	}

	return tuple!("trainFeatures", "testFeatures", "trainLabels", "testLabels", "unlabelled", "indices")
				 (loadFeatures(directory ~ "train_X.bin"),
				  loadFeatures(directory ~ "test_X.bin"),
				  loadLabels(directory ~ "train_y.bin"),
				  loadLabels(directory ~ "test_y.bin"),
				  loadFeatures(directory ~ "unlabeled_X.bin"),
				  loadIndices(directory ~ "fold_indices.txt"));
}


module dnnet.data.patches;

import std.algorithm;
import std.array;
import std.conv;
import std.format;
import std.stdio;
import std.typecons;

import imageformats;

auto loadPatches(string dir)
{
	alias T = float;

	T[][] features;
	T[][] labels = File(dir ~ "/info.txt", "r")
				  .byLineCopy()
				  .map!(x => [x.splitter.front.to!T])
				  .array();

	size_t numPatchSets = labels.length / (16 * 16);

	if(labels.length % (16 * 16) != 0)
	{
		numPatchSets++;
	}

	string[] imageList;

	for(size_t i = 0; i < numPatchSets; i++)
	{
		imageList ~= format("%s/patches%04d.bmp", dir, i);
	}

	foreach(imageName; imageList)
	{
		auto image = read_image(imageName, ColFmt.Y);
		size_t vol = 1024 * 1024;

		for(size_t p = 0; p < 16 * 16; p++)
		{
			auto patch = new T[64 * 64];

			for(size_t y = 0; y < 64; y++)
			{
				for(size_t x = 0; x < 64; x++)
				{
					auto py = p / 16;
					auto px = p % 16;

					patch[y * 64 + x] = cast(T)image.pixels[(py * 64 + y) * 1024 + (px * 64 + x)] / 128.0 - 1.0;
				}
			}

			features ~= patch;
		}
	}

	return tuple(features, labels);
}

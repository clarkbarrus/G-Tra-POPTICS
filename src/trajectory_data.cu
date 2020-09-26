/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include "trajectory_data.cuh"

#define DEBUG true

file_trajectory_data load_trajectory_data_from_file(std::string file_name)
{

	if (DEBUG)
	{
		std::cout << "Loading data from " << file_name << std::endl;
	}

	// First count the number of trajectories and number of points to allow for linear time vector construction
	std::ifstream first_input_file;
	first_input_file.open(file_name);

	// Error check input file stream
	if (!first_input_file.is_open())
	{
		std::cerr << "Input file unable to open" << std::endl;
	}

	std::string line;
	getline(first_input_file, line); // Throw away header;

	int cur_trajectory_number = -1;
	int num_trajectories = 0;
	int num_points = 0;

	// Read in each line of the file
	while (getline(first_input_file, line))
	{
		if (DEBUG)
		{
			std::cout << line << std::endl;
		}

		// For this file type, each point is a line. Trajectories described using trajectory number.
		num_points++;

		// Extract trajectory number from line.
		// Test input is of the form: trajectory_number, x, y, t

		// Parse input
		point point;
		char * cstr = new char [line.length()+1]; // Convert to line for tokenization. Not the best way to do this I know.
		std::strcpy(cstr, line.c_str());
		point.trajectory_number = std::stoi(strtok(cstr, ","));

		if (point.trajectory_number != cur_trajectory_number) {
			//We are on a new trajectory, increment trajectory count
			num_trajectories++;
			cur_trajectory_number = point.trajectory_number;
		}

		delete[] cstr;
	}

	first_input_file.close();

	/**
	 * We now have a count of the number of points and trajectories. Allocate arrays for points and trajectory indices.
	 */
	point* points = new point[num_points];
	trajectory_index* 	trajectories = new trajectory_index[num_trajectories];
	int point_index = 0;
	trajectory_index* cur_trajectory_index = trajectories;
	int cur_trajectory_length = 0;
	cur_trajectory_index->traj_start = points;
	cur_trajectory_number = 1;

	if (DEBUG)
	{
		std::cout << "File has " << num_points << " points and " << num_trajectories << " trajectories" << std::endl;
	}

	// Open file for reading and parsing
	std::ifstream input_file;
	input_file.open(file_name);

	// Error check input file stream
	if (!input_file.is_open())
	{
		std::cerr << "Input file unable to open" << std::endl;
	}

	getline(input_file, line); // Throw away header;

	// Read in each line of the file
	while (getline(input_file, line))
	{
		// Extract trajectory information from line.
		// Test input is of the form:
		// trajectory_number, x, y, t

		if (DEBUG)
		{
			std::cout << line << std::endl;
		}

		point *point = &points[point_index]; // Variable to read line information into

		// Parse each input line into point{points, trajectories, num_points, num_trajectories};
		char * cstr = new char [line.length()+1]; // Convert to line for tokenization. Not the best way to do this I know.
		std::strcpy(cstr, line.c_str());
		point->trajectory_number = std::stoi(strtok(cstr, ","));
		point->t = std::stod(strtok(NULL, ","));
		point->x = std::stod(strtok(NULL, ","));
		point->y = std::stod(strtok(NULL, ","));

		delete[] cstr;

		// Update point and trajectory index information

		if (point->trajectory_number != cur_trajectory_number)
		{
			// Update trajectory number
			cur_trajectory_index->traj_number = cur_trajectory_number;
			cur_trajectory_number = point->trajectory_number;

			// Update number of points in trajectory
			cur_trajectory_index->traj_length = cur_trajectory_length;
			cur_trajectory_length = 0;

			// Set up the next trajectory index
			cur_trajectory_index++;
			cur_trajectory_index->traj_start = point;
		}
		point_index++;
		cur_trajectory_length++;

	}

	input_file.close();

	// Add index information for last trajectory
	cur_trajectory_index->traj_number = cur_trajectory_number;
	cur_trajectory_index->traj_length = cur_trajectory_length;

	if(DEBUG)
	{
		std::cout << "Finished reading in a file with contents: " << std::endl;
		// Did the file read correctly?

		// Print all points{points, trajectories, num_points, num_trajectories};
		for (int i = 0; i < num_points; i++) {
			point point = points[i];
			std::cout << "Traj_num: " << point.trajectory_number << " x: " << point.x << " y: " << point.y << " t: " << point.t << std::endl;
		}

		// Print the trajectory indices we made
		for (int i = 0; i < num_trajectories; i++) {
			trajectory_index index = trajectories[i];
			std::cout << "TrajectoryID: "<< index.traj_number << " traj_length: " << index.traj_length << " traj_start*: " << index.traj_start << std::endl;
		}

		// Print metadata
		std::cout << "Number of points: " << num_points << std::endl;
		std::cout << "Number of trajectories: " << num_trajectories << std::endl;
	}

	file_trajectory_data data = {points, trajectories, num_points, num_trajectories};
	return data;
}

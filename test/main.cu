/*
 * main.cu
 *
 *  Created on: Sep 26, 2020
 *      Author: clark
 */

#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"
#include "trajectory_data.cuh"
#include "g_tra_poptics.cuh"
#include "strtree.cuh"

/**
 * This macro checks return value of the CUDA runtime call and exits
 * the application if the call failed.
 */
#define CUDA_CHECK_RETURN(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		fprintf(stderr, "Error %s at line %d in file %s\n",					\
				cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		exit(1);															\
	} }

/**
 *
 * Entry point for executing G-Tra-POPTICS
 */
int main(int argc, char **argv)
{
	std::string file_name = "data/testtrajectorydata.csv";

	// Load trajectory data from file
	file_trajectory_data trajectory_data = load_trajectory_data_from_file(file_name);

	// Preprocessing: build STR-tree index
	thrust::host_vector<strtree_line> lines = points_to_line_vector(trajectory_data.points, trajectory_data.trajectories,
			trajectory_data.num_points, trajectory_data.num_trajectories);

	// Build new trajectory index for vector "lines"
	thrust::host_vector<size_t> trajectory_start_indices(trajectory_data.num_trajectories);
	for (size_t i = 0, num_lines = 0; i < trajectory_data.num_trajectories; i++)
	{
		trajectory_start_indices[i] = num_lines;
		if (i < trajectory_data.num_trajectories - 1)
		{
			// Each trajectory has one fewer line than it had points.
			//num_lines += trajectory_data.trajectories[i + 1] - trajectory_data.trajectories[i] - 1;
		}
	}

	// Create index structure
	strtree strtree = cuda_create_strtree(lines);

	thrust::host_vector<strtree_offset_node> nodes = strtree.nodes;
	for(int i = 0; i < nodes.size(); i++)
	{
		strtree_offset_node node = nodes[i];
		std::cout << "Node " << i << ": num children=" << node.num << ", depth=" << node.depth << ", child_offset=" <<node.first_child_offset
			<< ", bbox.x1=" << node.boundingbox.x1 << ", bbox.x2=" << node.boundingbox.x2
			<< ", bbox.y1=" << node.boundingbox.y1 << ", bbox.y2=" << node.boundingbox.y2
			<< ", bbox.t1=" << node.boundingbox.t1 << ", bbox.t2=" << node.boundingbox.t2
			<< std::endl;
	}

//	/* Initialize variables for G-Tra-POPTICS execution */
//	// Number of CPU threads executing
//	int cpu_threads = 8;
//	// Maximum epsilon at which clusters are detected
//	double epsilon = 0.2;
//	// Specific epsilon for which to find clusters after minimum spanning trees are built
//	double epsilon_prime = 0.1;
//	// Minimum number of trajectories near a point for it to be considered a core point.
//	double min_num_trajectories = 2;
//
//	// Execute G-Tra-POPTICS on data file
//	g_tra_poptics(strtree, cpu_threads, epsilon, epsilon_prime, min_num_trajectories);

	/****** To run unit tests use this return ******/
	printf("Running unit tests from main.cu\n");
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();

	/****** To disable unit tests use this return ******/
//	return 0;
}




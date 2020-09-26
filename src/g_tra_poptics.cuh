/*
 * g_tra_poptics.cuh
 *
 *  Created on: Sep 2, 2020
 *      Author: clark
 */

#ifndef G_TRA_POPITCS_CUH_
#define G_TRA_POPITCS_CUH_

#include <vector>
#include "trajectory_data.cuh"
#include "strtree.cuh"

const int N = 10; // Linear interpolation granularity for st-distance calculations

// Struct for doing simplified vector computation
struct Point
{
    double x;
    double y;
};

// Entry point for the G_Tra_POPTICS algorithm
int g_tra_poptics(strtree strtree, int cpu_threads, double epsilon, double epsilon_prime, int min_num_trajectories);

// Calculates the spatio-temporal distance between two trajectories.
double stdistance_between_trajectories(thrust::device_vector<strtree_line> lines,
		size_t trajectory_p_offset, size_t trajectory_p_length,
		size_t trajectory_q_offset, size_t trajectory_q_length);

__device__ __host__
double stdistance_between_trajectory_and_mbb(strtree_line *lines, size_t trajectory_offset, size_t trajectory_length,
		strtree_rect &rectangle);

// Calculates the spatio-temporal distance between two trajectories.
__device__ __host__
double stdistance_between_trajectories(strtree_line *lines,
		size_t trajectory_p_offset, size_t trajectory_p_length,
		size_t trajectory_q_offset, size_t trajectory_q_length);

// A C++ program to check if two given line segments intersect
// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
__device__ __host__
bool doIntersect(Point p1, Point q1, Point p2, Point q2);

// Function to return the minimum distance
// between a line segment AB and a point E
// https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
__device__ __host__
double mindist_line_to_point(double a_x, double a_y, double b_x, double b_y, double e_x, double e_y);

// 2-dimensional euclidean distance
__device__ __host__
double distance(double x1, double y1, double x2, double y2);

__global__
void neighborhood_query_dfs_count(strtree tree, thrust::device_vector<size_t> trajectory_start_indices);

void neighborhood_query(strtree tree, thrust::device_vector<size_t> trajectory_start_indices, double epsilon);;

#endif /* G_TRA_POPITCS_CUH_ */

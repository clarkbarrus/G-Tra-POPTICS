/*
 * trajectory_data.cuh
 *
 *  Created on: Sep 2, 2020
 *      Author: clark
 */

#ifndef TRAJECTORY_DATA_CUH_
#define TRAJECTORY_DATA_CUH_

// Includes
#include <vector>
#include <string>

// Structs

// Trajectory data will be organized as a vector of these points. This will make transfer of data to GPU possible.
// Trajectory level information will be maintained in a trajectory index if necessary
struct point
{
	int trajectory_number;
	double x;
	double y;
	double t;
	__device__ __host__ point():trajectory_number(0),x(0.0),y(0.0),t(0.0){}
	__device__ __host__ point(int trajectory_number, double x, double y, double t):
			trajectory_number(trajectory_number), x(x), y(y), t(t){}
};

struct trajectory_index
{
	int 				traj_number;
	point* 				traj_start;
	int 				traj_length;
};

struct file_trajectory_data
{
	point* 				points;
	trajectory_index* 	trajectories;
	int 				num_points;
	int 				num_trajectories;
};

// Prototypes
file_trajectory_data load_trajectory_data_from_file(std::string file_name);


#endif /* TRAJECTORY_DATA_CUH_ */

/*
 * strtree.cu
 *
 *  Created on: Sep 13, 2020
 *      Author: Clark Barrus based on rtree code from github.com/phifaner/comdb/rtree.cu
 */

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <iostream>

#include "strtree.cuh"

#define DEBUG true

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

//__host__ __device__
//int contains(RTree_Rect *R, RTree_Point *P)
//{
//    register RTree_Rect *r = R;
//    register RTree_Point *p = P;
//
//    assert(r && p);
//
//    //printf("point: %llu, %lu, Rect: %llu, %llu, %lu, %lu\n",
//    //        p->x, p->y, r->left, r->right, r->top, r->bottom);
//
//    if (p->x < r->right && p->x > r->left
//            && p->y < r->bottom && p->y > r->top)
//        return 1;
//    else
//        return 0;
//}

__host__ __device__
inline void init_boundary(strtree_rect *bbox)
{
    bbox->x1 = DBL_MAX;
    bbox->x2 = -DBL_MAX;
    bbox->y1 = DBL_MAX;
    bbox->y2 = -DBL_MAX;
    bbox->t1 = DBL_MAX;
    bbox->t2 = -DBL_MAX;
}

//__host__ __device__
//inline void update_boundary(RTree_Rect *bbox, RTree_Rect *node_bbx)
//{
//    bbox->top = min(bbox->top, node_bbx->top);
//    bbox->bottom = max(bbox->bottom, node_bbx->bottom);
//    bbox->left = min(bbox->left, node_bbx->left);
//    bbox->right = max(bbox->right, node_bbx->right);
//
//    //printf("---node bbox: %llu, %llu, update: %llu, %llu\n",
//    //        node_bbx->left, node_bbx->right, bbox->left, bbox->right);
//
//}

__host__ __device__
inline void update_boundary(strtree_rect *bbox, strtree_rect *node_bbx)
{
	bbox->x1 = min(node_bbx->x1, bbox->x1);
	bbox->x2 = max(node_bbx->x2, bbox->x2);
	bbox->y1 = min(node_bbx->y1, bbox->y1);
	bbox->y2 = max(node_bbx->y2, bbox->y2);
	bbox->t1 = min(node_bbx->t1, bbox->t1);
	bbox->t2 = max(node_bbx->t2, bbox->t2);
}

__host__ __device__
inline void c_update_boundary(strtree_rect *bbox, strtree_line *p)
{
    bbox->x1 = min(p->line_boundingbox.x1, bbox->x1);
    bbox->x2 = max(p->line_boundingbox.x2, bbox->x2);
    bbox->y1 = min(p->line_boundingbox.y1, bbox->y1);
    bbox->y2 = max(p->line_boundingbox.y2, bbox->y2);
    bbox->t1 = min(p->line_boundingbox.t1, bbox->t1);
	bbox->t2 = max(p->line_boundingbox.t2, bbox->t2);

    //printf("x: %llu, bbox: %lu, %lu, %llu, %llu\n", p->x, bbox->top, bbox->bottom, bbox->left, bbox->right);
}

__host__ __device__
inline size_t get_node_length (
        const size_t i,
        const size_t len_level,
        const size_t previous_level_len,
        const size_t node_size)
{
    const size_t n = node_size;
    const size_t len = previous_level_len;
    const size_t final_i = len_level -1;

    // set lnum to len%n if it's the last iteration and there's a remainder, else n
    return ((i != final_i || len % n == 0) *n) + ((i == final_i && len % n != 0) * (len % n));
}

__global__
void create_level_kernel
		(
			strtree_offset_node *d_nodes,
			size_t parent_start_offset,
			size_t parent_end_offset,
			size_t child_start_offset,
			size_t child_end_offset,
			size_t depth
		)
{
	// Thread should execute for each node in d_parent_nodes
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; //Use grid stride loop strategy
	         i < parent_end_offset - parent_start_offset;
	         i += blockDim.x * gridDim.x)
	{
		strtree_offset_node node;
		node.first_child_offset = child_start_offset + i * STRTREE_NODE_SIZE;
		node.num = get_node_length(i, parent_end_offset - parent_start_offset, child_end_offset - child_start_offset, STRTREE_NODE_SIZE);
		node.depth = depth;

		// Update node's bbox
	    init_boundary(&node.boundingbox);
		#pragma unroll //TODO Consider data coalescing here
		for (size_t j = 0, jend = node.num; j != jend; ++j)
		{
			 // Using device references to actually access child node
			update_boundary(&node.boundingbox, &d_nodes[node.first_child_offset + j].boundingbox);

		}

		// Write resulting node
		d_nodes[i + parent_start_offset] = node;
	}
}

__global__
void create_leaves_kernel
		(
			strtree_line *d_lines,
			size_t lines_start_offset,
			size_t lines_end_offset,
			strtree_offset_node *d_nodes,
			size_t nodes_start_offset,
			size_t nodes_end_offset)
{
	// Thread should execute for each node in d_parent_nodes
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; //Use grid stride loop strategy
		         i < nodes_end_offset - nodes_start_offset;
		         i += blockDim.x * gridDim.x)
		{
			strtree_offset_node node;
			node.first_child_offset = lines_start_offset + i * STRTREE_NODE_SIZE;
			node.num = get_node_length(i, nodes_end_offset - nodes_start_offset, lines_end_offset - lines_start_offset, STRTREE_NODE_SIZE);
			node.depth = 0;

			// Update node's bbox
		    init_boundary(&node.boundingbox);
			#pragma unroll //TODO Consider data coalescing here
			for (size_t j = 0, jend = node.num; j != jend; ++j)
			{
				 // Using device references to actually access child node
				update_boundary(&node.boundingbox, &d_lines[node.first_child_offset + j].line_boundingbox);
			}

			// Write resulting node
			d_nodes[i + nodes_start_offset] = node;
		}
}

strtree cuda_create_strtree(thrust::host_vector<strtree_line> h_lines)
{
	// Skip sorting trajectories and lines for now. Not sure how to implement/how it would effect organization.
    //cuda_sort(&lines);

    // Calculate number of nodes in tree and tree height
    thrust::host_vector<size_t> level_offsets(1);
    level_offsets.push_back(0); //Tree leaves will start at 0
    size_t level_num_nodes = DIV_CEIL(h_lines.size(), STRTREE_NODE_SIZE);
    size_t total_num_nodes = 0 + level_num_nodes;

    while (level_num_nodes > 1)
    {
    	level_offsets.push_back(total_num_nodes);
    	level_num_nodes = DIV_CEIL(level_num_nodes, STRTREE_NODE_SIZE);
    	total_num_nodes += level_num_nodes;
    }

    const size_t tree_size = total_num_nodes;

	// Move lines to device
    thrust::device_vector<strtree_line> d_lines = h_lines;
    strtree_line *d_lines_ptr = (strtree_line*)thrust::raw_pointer_cast(d_lines.data());
    thrust::device_vector<strtree_offset_node> d_nodes(tree_size);
    strtree_offset_node *d_nodes_ptr = (strtree_offset_node*)thrust::raw_pointer_cast(d_nodes.data());

    // Leaves first
    size_t num_leaves = DIV_CEIL(h_lines.size(), STRTREE_NODE_SIZE);
    create_leaves_kernel<<< (num_leaves + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    					(d_lines_ptr, 0, d_lines.size(),
						 d_nodes_ptr, 0, num_leaves);

    // Build strtree from bottom
    size_t depth = 1;
    for(int i = 1; i < level_offsets.size(); i++)
    {
    	// This level has how many leaves?
    	if (i == level_offsets.size() - 1)
    	{
    		level_num_nodes = 1;
    	}
    	else
    	{
    		level_num_nodes = level_offsets[i+1] - level_offsets[i];
    	}

    	/**
    	 * For each node on this layer we want to
    	 * 1. Record the offset of the first node child
    	 * 2. Record the number of children
    	 * 3. Update the bounding box of the node based on each of the children
    	 */
    	create_level_kernel<<<(level_num_nodes + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>
    			(d_nodes_ptr, level_offsets[i], level_offsets[i]+level_num_nodes, level_offsets[i-1], level_offsets[i], depth);

    	depth++;
    }

    size_t root_offset = d_nodes.size() -1;

    strtree tree = {root_offset, d_nodes, d_lines};

//    if (DEBUG)
//	{
//		std::cout << "Root node at level " << depth << "create_strtree() returns:" << std::endl;
//		strtree_node node = *root;
//		std::cout << "Root node: num=" << node.num << ": depth=" << node.depth
//				<< ": bbox.x1=" << node.boundingbox.x1 << ": bbox.x2=" << node.boundingbox.x2
//				<< ": bbox.y1=" << node.boundingbox.y1 << ": bbox.y2=" << node.boundingbox.y2
//				<< ": bbox.t1=" << node.boundingbox.t1 << ": bbox.t2=" << node.boundingbox.t2 << std::endl;
//		for (int j = 0; j < node.num; j++)
//		{
//			strtree_node child_node = node.children[j];
//			std::cout << "    Child node " << j << ": num=" << child_node.num << ", depth=" << child_node.depth
//				<< ", bbox.x1=" << child_node.boundingbox.x1 << ", bbox.x2=" << child_node.boundingbox.x2
//				<< ", bbox.y1=" << child_node.boundingbox.y1 << ", bbox.y2=" << child_node.boundingbox.y2
//				<< ", bbox.t1=" << child_node.boundingbox.t1 << ", bbox.t2=" << child_node.boundingbox.t2
//				<< std::endl;
//		}
//	}

    return tree;
}


//strtree_leaf* cuda_create_leaves(strtree_lines *sorted_lines)
//{
//    const size_t len = sorted_lines->length;
//    const size_t num_leaf = DIV_CEIL(len, STRTREE_NODE_SIZE);
//
//    strtree_leaf  *d_leaves;
//    strtree_line  *d_lines;
//    int *d_ID;
//    int *d_Trajectory_Number;
//    strtree_rect *d_Line_BoundingBox;
//    short *d_Orientation;
//
//    if (DEBUG) { std::cout <<"Starting cudaMalloc for creating leaves" << std::endl; }
//
//    cudaMalloc( (void**) &d_leaves, num_leaf    * sizeof(strtree_leaf) );
//    cudaMalloc( (void**) &d_lines, len          * sizeof(strtree_line) );
//
//    if (DEBUG) { std::cout <<"Starting cudaMalloc for sorted_lines contents" << std::endl; }
//
//    // Move a copy of sorted_lines to the device so the device can see the data
//    if (DEBUG) { std::cout <<"Starting cudaMalloc for d_ID" << std::endl; }
//    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_ID, len * sizeof(int)));
//    if (DEBUG) { std::cout <<"Starting cudaMemcpy for d_ID" << std::endl; }
//    CUDA_CHECK_RETURN(cudaMemcpy(d_ID, sorted_lines->ID, len * sizeof(int), cudaMemcpyHostToDevice));
//    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_Trajectory_Number, len * sizeof(int)));
//    CUDA_CHECK_RETURN(cudaMemcpy(d_Trajectory_Number, sorted_lines->Trajectory_Number, len * sizeof(int), cudaMemcpyHostToDevice));
//    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_Line_BoundingBox, len * sizeof(strtree_rect)));
//    CUDA_CHECK_RETURN(cudaMemcpy(d_Line_BoundingBox, sorted_lines->Line_BoundingBox, len * sizeof(strtree_rect), cudaMemcpyHostToDevice));
//    CUDA_CHECK_RETURN(cudaMalloc( (void**) &d_Orientation, len * sizeof(short)));
//    CUDA_CHECK_RETURN(cudaMemcpy(d_Orientation, sorted_lines->Orientation, len * sizeof(short), cudaMemcpyHostToDevice));
//	//d_sorted_lines->length = sorted_lines->length;
//
//    if (DEBUG) { std::cout <<"Finished cudaMalloc and memcpy for sorted_lines sorted_lines contents" << std::endl; }
//
//    // Leaves on device will copy lines into here this host array and maintain pointers to positions in this array
//    strtree_line *lines = new strtree_line[len];
//
//    if (DEBUG) { std::cout <<"Launching create_leaves_kernel" << std::endl; }
//
//    create_leaves_kernel<<< (num_leaf + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
//        (d_leaves, d_lines, lines, d_ID, d_Trajectory_Number, d_Line_BoundingBox, d_Orientation, sorted_lines->length);
//
//    if (DEBUG) { std::cout <<"Finished create_leaves_kernel" << std::endl; }
//
//    strtree_leaf  *leaves = new strtree_leaf[num_leaf];
//
//    // copy points from device to host
//    cudaMemcpy(leaves, d_leaves, num_leaf   * sizeof(strtree_leaf), cudaMemcpyDeviceToHost);
//    cudaMemcpy(lines, d_lines, len        * sizeof(strtree_line), cudaMemcpyDeviceToHost);
//
//    cudaFree(d_leaves);
//    cudaFree(d_lines);
//    cudaFree(d_ID);
//    cudaFree(d_Trajectory_Number);
//    cudaFree(d_Line_BoundingBox);
//    cudaFree(d_Orientation);
//
//    if (DEBUG)
//	{
//    	std::cout << "Lines copied from device point to the following lines now on host" << std::endl;
//		for(int i = 0; i < len; i++)
//		{
//			strtree_line line = lines[i];
//			std::cout << "Line " << i << ": id=" << line.id << ", trajectory_number=" << line.trajectory_number
//					<< ", bbox.x1=" << line.line_boundingbox.x1 << ", bbox.x2=" << line.line_boundingbox.x2
//					<< ", bbox.y1=" << line.line_boundingbox.y1 << ", bbox.y2=" << line.line_boundingbox.y2
//					<< ", bbox.t1=" << line.line_boundingbox.t1 << ", bbox.t2=" << line.line_boundingbox.t2
//					<< ", orientation=" << line.orientation << std::endl;
//		}
//		std::cout << "Leaves generated on device after transfer back to host:" << std::endl;
//		for (int i = 0; i < num_leaf; i++) {
//			strtree_leaf leaf = leaves[i];
//			std::cout << "Leaf " << i << ": num=" << leaf.num << ": depth=" << leaf.depth
//					<< ": bbox.x1=" << leaf.boundingbox.x1 << ": bbox.x2=" << leaf.boundingbox.x2
//					<< ": bbox.y1=" << leaf.boundingbox.y1 << ": bbox.y2=" << leaf.boundingbox.y2
//					<< ": bbox.t1=" << leaf.boundingbox.t1 << ": bbox.t2=" << leaf.boundingbox.t2 << std::endl;
//    		for (int j = 0; j < leaf.num; j++)
//    		{
//    			strtree_line line = leaf.lines[j];
//    			std::cout << "    Child line " << j << ": id=" << line.id << ", trajectory_number=" << line.trajectory_number
//					<< ", bbox.x1=" << line.line_boundingbox.x1 << ", bbox.x2=" << line.line_boundingbox.x2
//					<< ", bbox.y1=" << line.line_boundingbox.y1 << ", bbox.y2=" << line.line_boundingbox.y2
//					<< ", bbox.t1=" << line.line_boundingbox.t1 << ", bbox.t2=" << line.line_boundingbox.t2
//					<< ", orientation=" << line.orientation << std::endl;
//    		}
//		}
//	}
//
//    return leaves;
//
//}

// V2.0 of points to lines. This is the iteration of the data structure using offsets and vector instead of arrays & pointers.
thrust::host_vector<strtree_line> points_to_line_vector(
		point* points, trajectory_index* trajectory_indices, int num_points, int num_trajectories)
{
	// Want to create an vector of strtree_line structs
	//	std::vector<strtree_line> lines

	// Each trajectory has 1 fewer lines than points.
	size_t num_lines = num_points - num_trajectories;

	thrust::host_vector<strtree_line> lines(num_lines);
	int id = 0; // Use line id as an index tracking which line we are on.

	point last_point = points[0];
	point this_point;
	for (int i = 1; i < num_points; i++) //TODO do this work using a GPU kernel
	{
		this_point = points[i];

		if(DEBUG)
		{
			std::cout << "Adding line between points:" << std::endl;
			std::cout << "Traj_num: " << last_point.trajectory_number << " x: " << last_point.x << " y: " << last_point.y << " t: " << last_point.t << std::endl;
			std::cout << "Traj_num: " << this_point.trajectory_number << " x: " << this_point.x << " y: " << this_point.y << " t: " << this_point.t << std::endl;
		}

		if (last_point.trajectory_number != this_point.trajectory_number)
		{
			// We are now on a new trajectory. Don't add another line
			last_point = this_point;
			continue;
		}

		strtree_line *line = &lines[id];

		// Create a line between last point and this point.
		line->id = id;
		line->trajectory_number = this_point.trajectory_number;
		line->line_boundingbox = points_to_bbox(last_point, this_point);
		line->orientation = points_to_orientation(last_point, this_point);

		// Set up for next execution of the for loop
		id++;
		last_point = this_point;
	}

	if (DEBUG)
	{
		std::cout << "Contents of vector<strtree_line> lines returned by points_to_line_vector()" << std::endl;
		for(int i = 0; i < num_lines; i++)
		{
			std::cout << "Line " << i << ": id=" << lines[i].id << ", trajectory_number=" << lines[i].trajectory_number
					<< ", bbox.x1=" << lines[i].line_boundingbox.x1 << ", bbox.x2=" << lines[i].line_boundingbox.x2
					<< ", bbox.y1=" << lines[i].line_boundingbox.y1 << ", bbox.y2=" << lines[i].line_boundingbox.y2
					<< ", bbox.t1=" << lines[i].line_boundingbox.t1 << ", bbox.t2=" << lines[i].line_boundingbox.t2
					<< ", orientation=" << lines[i].orientation << std::endl;
		}
	}

	return lines;
}

strtree_rect points_to_bbox(point p1, point p2)
{
	strtree_rect rect;
	rect.x1 = min(p1.x, p2.x);
	rect.x2 = max(p1.x, p2.x);
	rect.y1 = min(p1.y, p2.y);
	rect.y2 = max(p1.y, p2.y);
	rect.t1 = min(p1.t, p2.t);
	rect.t2 = max(p1.t, p2.t);
	return rect;
}

short points_to_orientation(point p1, point p2)
{
	/** Recall:
	 * orientation indicates how the line represented by the entry is represented by the bounding box above
	 * Since trajectories move forward in time, first point is always t1, second t2.
	 * 0: (x1,y1) to (x2,y2)
	 * 1: (x1,y2) to (x2,y1)
	 * 2: (x2,y1) to (x1,y2)
	 * 3: (x2,y2) to (x1,y1)
	 */
	if (p1.x < p2.x)
	{
		if (p1.y < p2.y)
		{
			return 0;
		}
		else
		{
			return 1;
		}
	}
	else
	{
		if (p1.y < p2.y)
		{
			return 2;
		}
		else
		{
			return 3;
		}
	}
}

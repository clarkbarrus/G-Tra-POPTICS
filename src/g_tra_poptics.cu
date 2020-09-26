/* *
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include "g_tra_poptics.cuh"


int g_tra_poptics(strtree strtree, int cpu_threads, double epsilon, double epsilon_prime, int min_num_trajectories)
{
	// Load trajectory data onto GPU

	// Divide data into subsets (uses either CPU multithreading OR GPU dynamic concurrency

	// Initialize global priority

	// For each CPU thread generate local MST's (need GPU version)

	// For each trajectory in data set
	// Mark as processed
	// Get neighbors of trajectory (TODO)
	// set_core_distance(tr, Ns, epsilon, min num of trs)
	// if tr.coreDistance != null
	// update local priority queue with edges to each neighbor
	// Until local priority queue empty
	// Process next edge in priority queue

	// Process next edge in priority queue:
	// Take minimum edge. Insert into global priority queue.
	// set core distance for this trajectory
	//

	// Find neighbors from global dataset
	// Add neighbors to local priority queue
	// Repeat until priority queue is empty

	return 0;
}

// Implementation of spatio temporal distance measure between a trajectory and a minimum bounding rectangle. This is a best effort
// reproduction of the algorithm described in Deng 2015 p. 555
__device__ __host__
double stdistance_between_trajectory_and_mbb(strtree_line *lines, size_t trajectory_offset, size_t trajectory_length,
		strtree_rect &rectangle)
{
	// For each line segment in the trajectory, find the minimum distance between each projection of the line segment and the rectangle.
	double distance_sum = 0.0;
	for(size_t line_offset = 0; line_offset < trajectory_length; line_offset++)
	{
		strtree_line line = lines[trajectory_offset + line_offset];
		strtree_rect bbox = line.line_boundingbox;
		double t1 = bbox.t1;
		double t2 = bbox.t2;

		// Use orientation to parse info about line segment
		/**
		 * orientation indicates how the line represented by the entry is represented by the bounding box above
		 * Since trajectories move forward in time, first point is always t1, second t2.
		 * 0: (x1,y1) to (x2,y2)
		 * 1: (x1,y2) to (x2,y1)
		 * 2: (x2,y1) to (x1,y2)
		 * 3: (x2,y2) to (x1,y1)
		 */
		double x1, x2, y1, y2;
		switch(line.orientation)
		{
			case 0:
				x1 = bbox.x1, x2 = bbox.x2, y1 = bbox.y1, y2 = bbox.y2;
				break;
			case 1:
				x1 = bbox.x1, x2 = bbox.x2, y1 = bbox.y2, y2 = bbox.y1;
				break;
			case 2:
				x1 = bbox.x2, x2 = bbox.x1, y1 = bbox.y1, y2 = bbox.y2;
				break;
			case 3:
				x1 = bbox.x2, x2 = bbox.x1, y1 = bbox.y2, y2 = bbox.y1;
				break;
		}

		/**
		 * Need to compute min distance for x-t plane and then the y-t plane.
		 *
		 * The line segment has four cases w.r.t. the rectangle
		 * 1. It passes through
		 * 2. It starts and stops in the same rectangle quadrant
		 * 3. It starts and stops in adjacent rectangle quadrants
		 * 4. It starts and stops in opposite rectangle quadrants
		 */

		/*** x-t plane ***/ // TODO Put plane code into a function instead of copying and pasting like a dunce.
		double xt_dist;

		// Does the line segment intersect the rectangle?
		if        (doIntersect({x1,t1}, {x2,t2}, {rectangle.x1, rectangle.t1}, {rectangle.x1, rectangle.t2})
				|| doIntersect({x1,t1}, {x2,t2}, {rectangle.x1, rectangle.t2}, {rectangle.x2, rectangle.t2})
				|| doIntersect({x1,t1}, {x2,t2}, {rectangle.x2, rectangle.t2}, {rectangle.x2, rectangle.t1})
				|| doIntersect({x1,t1}, {x2,t2}, {rectangle.x2, rectangle.t1}, {rectangle.x1, rectangle.t1}))
		{ // Intersect any side of the rectangle
			distance_sum += 0;
			continue;
		}

		// What quadrant is x1,t1 in? x2,t2?
		// Using standard quadrants with first in upper right, proceeding counter clockwise.
		// Considering quadrants relative to the center axis of the rectangle in each dimension
		short p1_quad;
		if (x1 > (rectangle.x1 + rectangle.x2) / 2)
		{
			if(t1 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p1_quad = 1;
			}
			else
			{
				p1_quad = 4;
			}
		}
		else
		{
			if(t1 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p1_quad = 2;
			}
			else
			{
				p1_quad = 3;
			}
		}

		short p2_quad;
		if (x2 > (rectangle.x1 + rectangle.x2) / 2)
		{
			if(t2 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p2_quad = 1;
			}
			else
			{
				p2_quad = 4;
			}
		}
		else
		{
			if(t2 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p2_quad = 2;
			}
			else
			{
				p2_quad = 3;
			}
		}

		if (p1_quad == p2_quad) // Case 2: Start/stop in same quadrant
		{
			// Calculate distance from line to quadrant corner
			switch(p1_quad)
			{
				case 1:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t2);
					break;
				case 2:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t2);
					break;
				case 3:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t1);
					break;
				case 4:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t1);
					break;
			}
		}
		else if (p1_quad % 4 == (p2_quad + 1) % 4
				|| p1_quad % 4 == (p2_quad - 1) % 4 ) // Case 3: Start/stop in adjacent quadrant
		{
			// Calculate distance from each quadrant corner and take min
			switch(p1_quad)
			{
				case 1:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t2);
					break;
				case 2:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t2);
					break;
				case 3:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t1);
					break;
				case 4:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t1);
					break;
			}

			switch(p2_quad)
			{
				case 1:
					xt_dist = min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t2));
					break;
				case 2:
					xt_dist =  min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t2));
					break;
				case 3:
					xt_dist =  min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t1));
					break;
				case 4:
					xt_dist =  min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t1));
					break;
			}
		}
		else	   // Case 4: Start/stop in opposite quadrant
		{
			// Calculate distance from opposite quadrant corners and take min
			switch((p1_quad + 1) % 4) // Shift quadrants counterclockwise to get "opposite" quadrants
			{
				case 1:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t2);
					break;
				case 2:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t2);
					break;
				case 3:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t1);
					break;
				case 0:
					xt_dist = mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t1);
					break;
			}

			switch((p2_quad + 1) % 4) // Shift quadrants counterclockwise to get "opposite" quadrants
			{
				case 1:
					xt_dist = min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t2));
					break;
				case 2:
					xt_dist =  min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t2));
					break;
				case 3:
					xt_dist =  min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x1, rectangle.t1));
					break;
				case 0:
					xt_dist =  min(xt_dist, mindist_line_to_point(x1, t1, x2, t2, rectangle.x2, rectangle.t1));
					break;
			}
		}

		/*** y-t plane ***/
		double yt_dist;

		// Does the line segment intersect the rectangle?
		if        (doIntersect({y1,t1}, {y2,t2}, {rectangle.y1, rectangle.t1}, {rectangle.y1, rectangle.t2})
				|| doIntersect({y1,t1}, {y2,t2}, {rectangle.y1, rectangle.t2}, {rectangle.y2, rectangle.t2})
				|| doIntersect({y1,t1}, {y2,t2}, {rectangle.y2, rectangle.t2}, {rectangle.y2, rectangle.t1})
				|| doIntersect({y1,t1}, {y2,t2}, {rectangle.y2, rectangle.t1}, {rectangle.y1, rectangle.t1}))
		{ // Intersect any side of the rectangle
			distance_sum += 0;
			continue; //TODO fix this, this code is wrong
		}

		// What quadrant is y1,t1 in? y2,t2?
		if (y1 > (rectangle.y1 + rectangle.y2) / 2)
		{
			if(t1 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p1_quad = 1;
			}
			else
			{
				p1_quad = 4;
			}
		}
		else
		{
			if(t1 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p1_quad = 2;
			}
			else
			{
				p1_quad = 3;
			}
		}

		if (y2 > (rectangle.y1 + rectangle.y2) / 2)
		{
			if(t2 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p2_quad = 1;
			}
			else
			{
				p2_quad = 4;
			}
		}
		else
		{
			if(t2 > (rectangle.t1 + rectangle.t2) / 2)
			{
				p2_quad = 2;
			}
			else
			{
				p2_quad = 3;
			}
		}

		if (p1_quad == p2_quad) // Case 2: Start/stop in same quadrant
		{
			// Calculate distance from line to quadrant corner
			switch(p1_quad)
			{
				case 1:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t2);
					break;
				case 2:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t2);
					break;
				case 3:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t1);
					break;
				case 4:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t1);
					break;
			}
		}
		else if (p1_quad % 4 == (p2_quad + 1) % 4
				|| p1_quad % 4 == (p2_quad - 1) % 4 ) // Case 3: Start/stop in adjacent quadrant
		{
			// Calculate distance from each quadrant corner and take min
			switch(p1_quad)
			{
				case 1:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t2);
					break;
				case 2:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t2);
					break;
				case 3:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t1);
					break;
				case 4:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t1);
					break;
			}

			switch(p2_quad)
			{
				case 1:
					yt_dist = min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t2));
					break;
				case 2:
					yt_dist =  min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t2));
					break;
				case 3:
					yt_dist =  min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t1));
					break;
				case 4:
					yt_dist =  min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t1));
					break;
			}
		}
		else	   // Case 4: Start/stop in opposite quadrant
		{
			// Calculate distance from opposite quadrant corners and take min
			switch((p1_quad + 1) % 4) // Shift quadrants counterclockwise to get "opposite" quadrants
			{
				case 1:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t2);
					break;
				case 2:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t2);
					break;
				case 3:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t1);
					break;
				case 0:
					yt_dist = mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t1);
					break;
			}

			switch((p2_quad + 1) % 4) // Shift quadrants counterclockwise to get "opposite" quadrants
			{
				case 1:
					yt_dist = min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t2));
					break;
				case 2:
					yt_dist =  min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t2));
					break;
				case 3:
					yt_dist =  min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y1, rectangle.t1));
					break;
				case 0:
					yt_dist =  min(yt_dist, mindist_line_to_point(y1, t1, y2, t2, rectangle.y2, rectangle.t1));
					break;
			}
		}

		// Line segment takes smallest distanc3e out of both projections
		distance_sum += min(xt_dist, yt_dist); // TODO See algoirthm 3 line 28 Deng 2015, this is wrong
	}

	return distance_sum;
}

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
// A C++ program to check if two given line segments intersect
__device__ __host__
bool onSegment(Point p, Point q, Point r)
{
    if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
        q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
       return true;

    return false;
}

// A C++ program to check if two given line segments intersect
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
__device__ __host__
int orientation(Point p, Point q, Point r)
{
    // See https://www.geeksforgeeks.org/orientation-3-ordered-points/
    // for details of below formula.
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);

    if (val == 0) return 0;  // colinear

    return (val > 0)? 1: 2; // clock or counterclock wise
}

// A C++ program to check if two given line segments intersect
// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
__device__ __host__
bool doIntersect(Point p1, Point q1, Point p2, Point q2)
{
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);

    // General case
    if (o1 != o2 && o3 != o4)
        return true;

    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;

    // p1, q1 and q2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;

    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;

     // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;

    return false; // Doesn't fall in any of the above cases
}

// Function to return the minimum distance
// between a line segment AB and a point E
// https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
__device__ __host__
double mindist_line_to_point(double a_x, double a_y, double b_x, double b_y, double e_x, double e_y)
{

    // vector ab
    double ab_x = b_x - a_x;
    double ab_y = b_y - a_y;

    // vector be
    double be_x = e_x - b_x;
    double be_y = e_x - b_y;

    // vector ae
    double ae_x = e_x - a_x;
    double ae_y = a_y - a_y;

    // Variables to store dot product
    double ab_be, ab_ae;

    // Calculating the dot product
    ab_be = (ab_x * be_x + ab_y * be_y);
    ab_ae = (ab_x * ae_x + ab_y * ae_y);

    // Minimum distance from
    // point E to the line segment
    double ans = 0;

    // Case 1
    if (ab_be > 0) {

        // Finding the magnitude
        double y = e_y - b_y;
        double x = e_x - b_x;
        ans = sqrt(x * x + y * y);
    }

    // Case 2
    else if (ab_ae < 0) {
        double y = e_y - a_y;
        double x = e_x - a_x;
        ans = sqrt(x * x + y * y);
    }

    // Case 3
    else {

        // Finding the perpendicular distance
        double x1 = ab_x;
        double y1 = ab_y;
        double x2 = ae_x;
        double y2 = ae_y;
        double mod = sqrt(x1 * x1 + y1 * y1);
        ans = abs(x1 * y2 - y1 * x2) / mod;
    }
    return ans;
}

// 2-dimensional euclidean distance
__device__ __host__
double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Given two interpolated line segments (they should line up temporally)
// computes the spatio temporal distance. Based on formulas in Deng 2015 (pp. 552-553)
__device__ __host__
double line_segment_stdistance(strtree_rect p, strtree_rect q)
{
	// t_k is p.t1, t_{k+1} is q.t2

	double A = (q.x2 - q.x1 - p.x2 + p.x1) * (q.x2 - q.x1 - p.x2 + p.x1)
			+ (q.y2 - q.y1 - p.y2 + p.y1) * (q.y2 - q.y1 - p.y2 + p.y1);
	double B = 2.0 * ((q.x2 - q.x1 - p.x2 + p.x1) * (q.x1 - p.x1)
			+ (q.y2 - q.y1 - p.y2 + p.y1) * (q.y1 - p.y1));
	double C = (q.x1 - p.x1) * (q.x1 - p.x1) + (q.y1 - p.y1) * (q.y1 - p.y1);
	double a = A / ((p.t2 - p.t1) * (p.t2 - p.t1));
	double b = B / (p.t2 - p.t1) - 2 * A * p.t2 / ((p.t2 - p.t1) * (p.t2 - p.t1));
	double c = A * p.t1 * p.t1 / ((p.t2 - p.t1) * (p.t2 - p.t1)) - B * p.t1 / (p.t2 - p.t1) + C;

	return (sqrt(a * p.t1 * p.t1 + b * p.t1 + c) + sqrt(a * p.t2 * p.t2 + b * p.t2 + c)) * (p.t2 - p.t1);
}

// Calculates the spatio-temporal distance between two trajectories.
__device__ __host__
double stdistance_between_trajectories(strtree_line *lines,
		size_t trajectory_p_offset, size_t trajectory_p_length,
		size_t trajectory_q_offset, size_t trajectory_q_length)
{
	// Calculate temporal overlap between trajectories
	strtree_line p_line = lines[trajectory_p_offset];
	size_t p_index = 0;
	strtree_line q_line = lines[trajectory_q_offset];
	size_t q_index = 0;
	double period_start;
	double period_end;

	// Set the start of the overlap period
	if(p_line.line_boundingbox.t1 > q_line.line_boundingbox.t1)
	{
		period_start = q_line.line_boundingbox.t1;
	}
	else
	{
		period_start = p_line.line_boundingbox.t1;
	}

	// Find the end of the overlap period.
	while (p_index < trajectory_p_length && q_index < trajectory_q_length)
	{
		if (p_line.line_boundingbox.t2 > q_line.line_boundingbox.t2)
		{
			q_index++;
			if (q_index == trajectory_q_length)
			{
				period_end = q_line.line_boundingbox.t2;
			}
			else
			{
				q_line = lines[trajectory_q_offset + q_index];
			}
		}
		else
		{
			p_index++;
			if (p_index == trajectory_p_length)
			{
				period_end = p_line.line_boundingbox.t2;
			}
			else
			{
				p_line = lines[trajectory_p_offset + p_index];
			}
		}
	}

	// period_start and period_end represent when the trajectories overlap.
	// if period_start >= period_end then there is no trajectory overlap.
	if (period_start >= period_end)
	{
		return DBL_MAX;
	}

	// Linearly interpolate trajectories based on overlap

	// Calculate the sum of each minimum distance between line segments in the linear interpolation
	double delta_t = (period_end - period_start) / (double) N;
	p_index = 0;
	q_index = 0;
	double distance_sum = 0;
	for (int k = 0; k < N - 1; k++)
	{
		// Calculate the minimum distance between linearly interpolated lines between times t_k and t_{k+1}
		double t_k = period_start + k * delta_t;
		double t_kp1 = t_k + delta_t;

		// Store interpolated lines using rect data structure
		strtree_rect interpolated_p_line;
		strtree_rect interpolated_q_line;

		// Use interpolation to calculate points at t_k for each trajectory
		while (lines[trajectory_p_offset + p_index].line_boundingbox.t2 < t_k)
		{
			p_index++; // TODO should not go out of bounds. There could be a mistake tho.
		}

		while (lines[trajectory_q_offset + q_index].line_boundingbox.t2 < t_k)
		{
			q_index++; // TODO should not go out of bounds. There could be a mistake tho.
		}

		// Calculate linear interpolation for first point
		p_line = lines[trajectory_p_offset + p_index];
		interpolated_p_line.x1 = (abs(p_line.line_boundingbox.x2 - p_line.line_boundingbox.x1)*(t_k - p_line.line_boundingbox.t1)/(p_line.line_boundingbox.t2 - p_line.line_boundingbox.t1));
		interpolated_p_line.y1 = (abs(p_line.line_boundingbox.y2 - p_line.line_boundingbox.y1)*(t_k - p_line.line_boundingbox.t1)/(p_line.line_boundingbox.t2 - p_line.line_boundingbox.t1));
		interpolated_p_line.t1 = t_k;

		q_line = lines[trajectory_q_offset + q_index];
		interpolated_q_line.x1 = (abs(q_line.line_boundingbox.x2 - q_line.line_boundingbox.x1)*(t_k - q_line.line_boundingbox.t1)/(q_line.line_boundingbox.t2 - q_line.line_boundingbox.t1));
		interpolated_q_line.y1 = (abs(q_line.line_boundingbox.y2 - q_line.line_boundingbox.y1)*(t_k - q_line.line_boundingbox.t1)/(q_line.line_boundingbox.t2 - q_line.line_boundingbox.t1));
		interpolated_q_line.t1 = t_k;

		// Use interpolation to calculate points at t_{k+1} for each trajectory;
		while (lines[trajectory_p_offset + p_index].line_boundingbox.t2 < t_kp1)
		{
			p_index++; // TODO should not go out of bounds. There could be a mistake tho.
		}

		while (lines[trajectory_q_offset + q_index].line_boundingbox.t2 < t_kp1)
		{
			q_index++; // TODO should not go out of bounds. There could be a mistake tho.
		}

		// Calculate linear interpolation for second point
		p_line = lines[trajectory_p_offset + p_index];
		interpolated_p_line.x2 = (abs(p_line.line_boundingbox.x2 - p_line.line_boundingbox.x1)*(t_kp1 - p_line.line_boundingbox.t1)/(p_line.line_boundingbox.t2 - p_line.line_boundingbox.t1));
		interpolated_p_line.y2 = (abs(p_line.line_boundingbox.y2 - p_line.line_boundingbox.y1)*(t_kp1 - p_line.line_boundingbox.t1)/(p_line.line_boundingbox.t2 - p_line.line_boundingbox.t1));
		interpolated_p_line.t2 = t_kp1;

		q_line = lines[trajectory_q_offset + q_index];
		interpolated_q_line.x2 = (abs(q_line.line_boundingbox.x2 - q_line.line_boundingbox.x1)*(t_kp1 - q_line.line_boundingbox.t1)/(q_line.line_boundingbox.t2 - q_line.line_boundingbox.t1));
		interpolated_q_line.y2 = (abs(q_line.line_boundingbox.y2 - q_line.line_boundingbox.y1)*(t_kp1 - q_line.line_boundingbox.t1)/(q_line.line_boundingbox.t2 - q_line.line_boundingbox.t1));
		interpolated_q_line.t2 = t_kp1;

		// Compute distance between these interpolated lines segments

		// Add interpolation to running sum
		distance_sum += line_segment_stdistance(interpolated_p_line, interpolated_q_line);
	}

	return distance_sum;
}

// Neighborhood query uses shared memory to maintain stack. Make sure to allocate this memory in the kernel call.
__global__
void neighborhood_query_dfs_count(strtree_line *lines, size_t num_lines, strtree_offset_node *nodes, size_t root_offset,
		size_t *trajectory_start_indices, size_t num_trajectories, double epsilon, int *count)
{
	// Check that block memory won't overflow with stacks
	// Each thread will use block memory = height of tree * (stack_element==short visited_children + size_t node_index)
	// Total memory should be = threads_per_block(1024)
	// Block should have maximum shared memory of 48kb
	// Should work for indicices up to a depth of ~4??? (46.875 Bytes of memory per GPU thread)
	short depth = nodes[root_offset].depth + 2;
	assert(48000 == blockDim.x * ((depth * (sizeof(short) + sizeof(size_t)))));
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
	         i < num_trajectories;
	         i += blockDim.x * gridDim.x)
	{
		// Allocate array for stack in shared memory

		// Local data (should go in registers?)
		short stack_pos = 0;
		size_t previous_trajectory_checked_index = UINT_MAX;
		size_t thread_trajectory_offset = trajectory_start_indices[i];
		size_t trajectory_length;
		if (i != num_trajectories)
		{
			trajectory_length = trajectory_start_indices[i + 1] - trajectory_start_indices[i];
		}
		else
		{
			trajectory_length = trajectory_start_indices[i] - num_lines;
		}

		extern __shared__ char s[];// Might need to do char
		size_t *node_index = (size_t*) s[threadIdx.x * depth * (sizeof(short) + sizeof(size_t))];
		short *visited_children = (short*)&node_index[depth];

		// Find all hits of trajectory i in the str tree.

		// Starting form the root, while the stack remains not empty.
		while(stack_pos > 0)
		{
			strtree_offset_node cur_node = nodes[node_index[stack_pos]];
			short cur_visited_children = visited_children[stack_pos];

			if (cur_visited_children >= STRTREE_NODE_SIZE)
			{
				// This node is complete.
				// Pop off top of stack and start next loop
				stack_pos--;
			}

			if (cur_node.depth == 0)
			{
				// We are currently at a leaf node, children are actually just lines.
				// Check if current line belongs to previously processed trajectory, if not check min distance
				if (lines[cur_node.first_child_offset + cur_visited_children].trajectory_number == previous_trajectory_checked_index)
				{
					// We have already checked the trajectory related to this line segment. Do nothing!
					visited_children[stack_pos]++;
				}
				else
				{
					previous_trajectory_checked_index = lines[cur_node.first_child_offset + cur_visited_children].trajectory_number;
					// TODO Should use precomputed table of neighbor values
					size_t trajectory_offset = trajectory_start_indices[previous_trajectory_checked_index];
					size_t cur_trajectory_length;
					if (trajectory_offset != num_trajectories)
					{
						cur_trajectory_length = trajectory_start_indices[previous_trajectory_checked_index + 1] - trajectory_start_indices[previous_trajectory_checked_index];
					}
					else
					{
						cur_trajectory_length = trajectory_start_indices[previous_trajectory_checked_index] - num_lines;
					}
					if (epsilon > stdistance_between_trajectories(lines, thread_trajectory_offset, trajectory_length, trajectory_offset,  cur_trajectory_length))
					{
						// Save value or increment as necessary.
						count[i]++;
					}
					visited_children[stack_pos]++;
				}
			}
			else
			{
				// Is the of the child MBB less than epsilon?
				if (epsilon > stdistance_between_trajectory_and_mbb(lines, thread_trajectory_offset, trajectory_length,
						nodes[cur_node.first_child_offset + cur_visited_children].boundingbox))
				{
					// In the neighborhood push the child onto the stack, and start visiting!
					visited_children[stack_pos]++;
					stack_pos++;
					node_index[stack_pos] = cur_node.first_child_offset + cur_visited_children;
					visited_children[stack_pos] = 0;
				}
				else
				{
					// Not in the neighborhood, increment number of children visited
					visited_children[stack_pos]++;
				}
			}
		}
	}
}

void neighborhood_query(strtree tree, thrust::device_vector<size_t> trajectory_start_indices, double epsilon)
{
	//int count[trajectory_start_indices.size()];
	// Run neighborhood query to get a count of results for each trajectory
	//neighborhood_query_dfs_count(tree.lines.pointer, tree.lines.size(), tree.nodes.pointer, tree.root_offset, trajectory_start_indices.pointer, trajectory_start_indices.size(), epsilon, int *count);

	// Allocate memory for results
	// For each count, allocate that many spaces.
	// Transfer to GPU

	// Run query again to actually get results
}


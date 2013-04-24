/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/

#ifndef TRAJECTORY_H_
#define TRAJECTORY_H_

#include <vector>
#include <opencv/cv.h>

/**
 * @author Clemens Korner
 */

namespace tld
{

/**
 * Detection trajectory; a line which associates the last detections.
 * The data is stored in 2 "shift registers"
 */
class Trajectory
{
public:
	/**
	 * Constructor; don't forget to call init(length) afterwards
	 */
	Trajectory();

	/**
	 * Destructor
	 */
	~Trajectory();

	/**
	 * Initializes the trajectory.
	 * @param length specifies the number of the last frames which are
	 * considered by the trajectory
	 */
	void init(std::size_t length);

	/**
	 * Adds a new point to the trajectory.
	 * If you want to add a "place holder" point just add a cvPoint(-1, -1)
	 * and cvScalar(-1 , -1, -1)
	 * @param point point which specifies the location of the point
	 * @param color defines the color (confidence) of the point
	 */
	void addPoint(CvPoint point, CvScalar color);

	/**
	 * Draws the trajectory into an image.
	 * @param image image to draw
	 */
	void drawTrajectory(IplImage * image);
private:
	std::size_t m_length; ///< number of the last frames which are considered by the trajectory
	std::vector<CvPoint> m_track_positions; ///< vector containing the locations of the points
	std::vector<CvScalar> m_track_colors; ///< vector containing the colors of the points
};

}

#endif /* TRAJECTORY_H_ */

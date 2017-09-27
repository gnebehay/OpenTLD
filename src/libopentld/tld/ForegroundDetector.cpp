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
/*
 * ForegroundDetector.cpp
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#include "ForegroundDetector.h"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

namespace tld
{

ForegroundDetector::ForegroundDetector()
{
    fgThreshold = 16;
    minArea = 0;
}

ForegroundDetector::~ForegroundDetector()
{
}

void ForegroundDetector::release()
{
}

void ForegroundDetector::nextIteration(const Mat &img)
{
    if(bgImg.empty())
    {
        return;
    }

    Mat absImg = Mat(img.cols, img.rows, img.type());
    Mat threshImg = Mat(img.cols, img.rows, img.type());
	std::vector<std::vector<Point> > contours;
	std::vector<std::vector<Point> > selected_contours;
	std::vector<Vec4i> hierarchy;

    absdiff(bgImg, img, absImg);
    threshold(absImg, threshImg, fgThreshold, 255, CV_THRESH_BINARY);

	findContours(absImg, contours, hierarchy,
					CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    for(size_t i = 0; i < contours.size(); i++)
		if(contourArea(contours[i]) > minArea)
			selected_contours.push_back(contours[i]);

    vector<Rect>* fgList = detectionResult->fgList;
    fgList->clear();

    for(size_t i = 0; i < selected_contours.size(); i++)
    {
        std::vector<Point> contour = selected_contours[i];
        Rect rect = boundingRect(contour);
        fgList->push_back(rect);
    }

}

bool ForegroundDetector::isActive()
{
    return !bgImg.empty();
}

} /* namespace tld */

#ifndef _ADAS_LANE_LOCATION_HPP_
#define _ADAS_LANE_LOCATION_HPP_

#include "opencv2/core.hpp"


class LaneLocate
{
public:

	LaneLocate(const cv::Size imageSize, const cv::Mat road2im);

	int operator()(cv::Vec4f line, cv::Mat image, float shift);

	int operator()(cv::Vec4f left, cv::Vec4f rght, cv::Mat image, float shift);

protected:

	cv::Scalar aveBGR;
	cv::Mat road2im;
	cv::Rect rect;
};


int LocateLane(cv::Vec4f line, cv::Mat image);


#endif//_ADAS_LANE_LOCATION_HPP_

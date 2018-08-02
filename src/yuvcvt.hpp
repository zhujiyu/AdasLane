#ifndef _SJS_COLOR_YUV_CVT_HPP_
#define _SJS_COLOR_YUV_CVT_HPP_

#include "opencv2/core.hpp"


void SubYUV422Image(const cv::Mat src, cv::Mat &dst, const cv::Rect rect);

void cvtYUV422toYellowGray(const cv::Mat src, cv::Mat &gray, const cv::Rect rect, cv::Mat &subsrc);

void cvtYUV422toYellowGray(const cv::Mat src, cv::Mat &gray);

void cvtYUV422toBGR(const cv::Mat src, cv::Mat &bgr);

void cvtYUV422toBGR(const cv::Mat src, cv::Mat &bgr, const cv::Rect rect, cv::Mat &subsrc);


void cvtBGR2YellowGray(const cv::Mat src, cv::Mat &gray, const cv::Rect rect);


#endif//_SJS_COLOR_YUV_CVT_HPP_

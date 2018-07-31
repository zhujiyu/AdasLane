#ifndef _SJS_COLOR_YUV_CVT_HPP_
#define _SJS_COLOR_YUV_CVT_HPP_


void cvtYUV422toYellowGray(const cv::Mat src, cv::Mat &gray, const cv::Rect rect);

void cvtBGR2YellowGray(const cv::Mat src, cv::Mat &gray, const cv::Rect rect);

void cvtYUV422toBGR(const cv::Mat src, cv::Mat &bgr, const cv::Rect rect);


void cvtYUV422toBGR(cv::Mat &bgr, const uchar* src_data, const cv::Size size, 
		size_t stripe = 0);

cv::Mat cvtYUV422toBGR(const uchar* src_data, const cv::Size size);

cv::Mat cvtYUV422toBGR(uchar *src_data, const cv::Size size,
		const cv::Rect rect);

cv::Mat cvtYUV422toYellowGray(const uchar* src_data, const cv::Size size,
		const size_t rowStep = 1);

cv::Mat cvtYUV422toYellowGray(uchar *src_data, const cv::Size size,
		const cv::Rect rect, const size_t rowStep = 1);

cv::Mat cvtBGR2YeloowGray(const cv::Mat src);

cv::Mat cvtBGR2YeloowGray(const cv::Mat src, const cv::Rect rect,
		const int rowStep = 1);

#endif//_SJS_COLOR_YUV_CVT_HPP_

#include "opencv2/highgui.hpp"

#include "locate.hpp"
#include "detect.hpp"
#include "track.hpp"

LaneLocate::LaneLocate(const cv::Size imageSize, const cv::Mat road2im):
		road2im(road2im), rect(DetectRegion(imageSize, road2im))
{}

inline void CalcLineByTwoPts(float &k, float &b, const cv::Point2f pt0,
		const cv::Point2f pt1)
{
    float dx = pt0.x - pt1.x;
    float dy = pt0.y - pt1.y;

    k = dx / dy;
    b = pt0.x - pt0.y * k;
}

inline void CalcPtsByLine(cv::Point2f &pt0, cv::Point2f &pt1,
		const cv::Vec4f src_line, const float step)
{
	float k = src_line[0]/src_line[1], b = -k*src_line[3] + src_line[2];
	float y0 = src_line[3] - step, y1 = src_line[3] + step;
    pt0 = cv::Point2f(k*y0+b, y0);
    pt1 = cv::Point2f(k*y1+b, y1);
}

void TransLine(float &k, float &b, cv::Vec4f line, cv::Mat tran, float step)
{
	std::vector<cv::Point2f> rts(2), pts(2);
    CalcPtsByLine(pts[0], pts[1], line, step);
    cv::perspectiveTransform(pts, rts, tran);
    CalcLineByTwoPts(k, b, rts[0], rts[1]);
}

int LaneLocate::operator()(cv::Vec4f line, cv::Mat image, float shift)
{
	float kl, kr, bl, br, x0 = line[2];
	cv::Mat mask(rect.size(), CV_8UC1);

	line[2] = x0 - shift;
	TransLine(kl, bl, line, road2im, 1000);
	line[2] = x0 + shift;
	TransLine(kr, br, line, road2im, 1000);

//	cv::Vec4f rsline = line, imline1, imline2;

//	rsline[2] = line[2] - shift;
//	TransLine(rsline, imline1, road2im, 1000);

//	rsline[2] = line[2] + shift;
//	TransLine(rsline, imline2, road2im, 1000);

//	const float k = line[0]/line[1], b = -k*(line[3]) + line[2];
//	float k = src_line[0]/src_line[1], b = -k*src_line[3] + src_line[2];


	mask.setTo(0);
	bl += rect.y * kl;
	br += rect.y * kr;

	for( int r = 0; r < rect.height; r++ )
	{
		float xl = MAX(0, r*kl + bl), xr = MIN(r*kr + br, rect.width);
		uchar* md = mask.ptr<uchar>(r);
		if( xr <= xl )
			continue;
		memset(md + (int)round(xl), 255, (int)round(xr - xl));
	}

	cv::imshow("mask", mask);

	return 0;
}

#ifndef _LANE_DETECT_HPP_SJS_ZJY_
#define _LANE_DETECT_HPP_SJS_ZJY_

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#define MAX_DISPARITY_PIXELS       128
#define MAX_SCAN_BLOCK_COUNT      1028  ///< 最多处理1028个块，但实际不一定有这么多
#define MAX_SCAN_IMAGE_ROWS        256  ///< 取图像中间部分，最多256行，实际也可能小于256
#define DEFAULT_BLOCK_SIZE           8


struct EdgeLine
{
	int left, rght, row, index;
	EdgeLine *next;

	EdgeLine(const int left, const int rght, const int row):
		left(left), rght(rght), row(row), index(0), next(0)
	{}
};


class LaneBlock
{
public:

	int index, pointCnt, lineCnt;
	float angle, slope;

	cv::Vec4f center_line;
	cv::Vec4f left_line, rght_line;
	EdgeLine *head, *tear;
	std::vector<cv::Vec3i> vertexs; ///< cv::Vec3i( left, right, row )

	LaneBlock(): index(0), pointCnt(0), lineCnt(0),
			angle(0), slope(0), head(0), tear(0)
	{}

	LaneBlock(const int idx): index(idx), pointCnt(0), lineCnt(0),
			angle(0), slope(0), head(0), tear(0)
	{}

	inline void Show(cv::Mat image, const int color_idx) const;

	friend inline std::ostream& operator<< (std::ostream& out,
			const LaneBlock &block);
};


struct DetectParam
{
	int stripe;
	int max_interv;     ///< 最大行距，大于此值的，将不归为一个block
	int max_dx;         ///< 中心点x坐标的差，这里是差的两倍，左右端点x坐标的和的差
	int max_dw, min_dw; ///< 两条短线的宽度的差值的最大值和最小值

	DetectParam(const int step): stripe(step), max_interv(4*step),
			max_dx(8*step), max_dw(4*step), min_dw(-step)
	{}

	inline bool CheckSame(const EdgeLine &pline, const EdgeLine &cline)
	{
		const int dx2 = pline.left + pline.rght - cline.left - cline.rght;
		const int dw  = cline.rght - cline.left - pline.rght + pline.left;
		return abs(dx2) <= max_dx && dw <= max_dw && dw >= min_dw;
	}

	inline bool CheckSame(const EdgeLine &pline, const EdgeLine &cline,
			const int prev_row, const int curr_row)
	{
		return curr_row - prev_row <= max_interv && CheckSame(pline, cline);
	}
};


class LaneBlockDetect
{
public:

	virtual ~LaneBlockDetect(){}

	LaneBlockDetect(): rowStep(1) {}

	static cv::Ptr<LaneBlockDetect> CreateDetect(const float thres = 120, const float min_lds = 6);

	static cv::Ptr<LaneBlockDetect> CreateDetect(const DetectParam *step,
			const float thres = 120, const float min_lds = 6);


	const std::vector<LaneBlock*> &GetLaneBlocks() const { return lane_blocks; }

	const cv::Rect &GetDetectRect() const { return rect; }

	const DetectParam *GetRowStep() const { return &rowStep; }


	virtual void SetStripe(DetectParam *step) = 0;

	virtual void SetDetectRegion(const cv::Size imageSize, const cv::Mat road2image,
			const float dist1 = 350, const float dist2 = 5000) = 0;

	virtual void SetDetectRegion(const cv::Size imageSize, const int bottom, const int top) = 0;

	virtual void operator()(const cv::Mat1s edgeImg) = 0;

	virtual void Show(cv::Mat dspImage, const uchar *flags = 0) const = 0;

	virtual void Show(cv::Mat dspImage, const int block_idx,
			const int color = 10) const = 0;

protected:

	DetectParam rowStep;
	cv::Rect rect;
	std::vector<LaneBlock*> lane_blocks;
};


cv::Rect DetectRegion(const cv::Size imageSize, const cv::Mat road2image,
		const float dist1 = 350, const float dist2 = 5000);

void GenEdgeImage(const cv::Mat src, const cv::Size size, cv::Mat1s &edge);

inline void DrawSolidLine(cv::Mat image, const cv::Vec4f line,
		const cv::Scalar color = cv::Scalar(0, 255, 255),
		const int width = 1, const cv::Point2f detla = cv::Point2f(0, 0))
{
	if( fabs(line[1]) < 0.001 )
		return;
	const float k = line[0]/line[1], b = -k*(line[3]+detla.y) + line[2]+detla.x;
	cv::line(image, cv::Point(b, 0), cv::Point(k*image.rows+b, image.rows),
			color, width);
}

inline void DrawDottedLine(cv::Mat image, const cv::Vec4f line,
		const cv::Scalar color = cv::Scalar(0, 255, 255),
		const int width = 1, const float start = 0.5,
		const cv::Point2f detla = cv::Point2f(0, 0))
{
	if( fabs(line[1]) < 0.001 )
		return;

	const float k = line[0]/line[1], b = -k*(line[3]+detla.y) + line[2]+detla.x;
	const float y0 = image.rows *.5, x0 = k*y0 + b;
	const float sx = image.rows *line[0] * .05, sy = image.rows *line[1] * .05;

	const float x1 = sx*start, y1 = sy*start;
	const cv::Point center(x0, y0), delta(x1, y1);

	for( int i = 0; i < 16; i += 2 )
	{
		float x = sx*i + x1, y = sy*i + y1;
		cv::Point p1(x, y), p2(x + sx, y + sy);
		cv::line(image, center+delta+p1, center+delta+p2, color, width);
		cv::line(image, center-delta-p1, center-delta-p2, color, width);
	}
}

inline void DrawDottedCircle(cv::Mat image, const float k, const float b,
		const cv::Scalar color, const int radius, const int width,
		const float start = 0.5, const cv::Point2f detla = cv::Point2f(0, 0))
{
	const float y0 = image.rows*.5, x0 = k*y0 + b;
	const float sx = k * 30, sy = 30;
	const float x1 = sx*start, y1 = sy*start;
	const cv::Point center(x0, y0), delta(x1, y1);

	for( int i = 0; i < 8; i++ )
	{
		float x = sx*i + x1, y = sy*i + y1;
		cv::circle(image, center+delta+cv::Point(x, y), radius, color, width);
		cv::circle(image, center-delta-cv::Point(x, y), radius, color, width);
	}
}

inline void DrawDottedCircle(cv::Mat image, const cv::Vec4f line,
		const cv::Scalar color, const int radius, const int width,
		const float start = 0.5, const cv::Point2f detla = cv::Point2f(0, 0))
{
	if( fabs(line[1]) < 0.001 )
		return;

	const float k = line[0]/line[1], b = -k*(line[3]+detla.y) + line[2]+detla.x;
	const float y0 = image.rows *.5, x0 = k*y0 + b;
	const float sx = image.rows *line[0] * .1, sy = image.rows *line[1] * .1;

	const float x1 = sx*start, y1 = sy*start;
	const cv::Point center(x0, y0), delta(x1, y1);

	for( int i = 0; i < 8; i++ )
	{
		float x = sx*i + x1, y = sy*i + y1;
		cv::circle(image, center+delta+cv::Point(x, y), radius, color, width);
		cv::circle(image, center-delta-cv::Point(x, y), radius, color, width);
	}
}

inline float line_distance(const cv::Vec4f line1, const cv::Vec4f line2)
{
	float dx = line1[2] - line2[2], dy = line1[3] - line2[3];
	float d1 = (line1[1]*dx - line1[0]*dy);
	float d2 = (line2[1]*dx - line2[0]*dy);
	return fabs(d1) + fabs(d2);
}

inline std::ostream& operator<< (std::ostream& out, const EdgeLine &line)
{
	out<<"["<<line.row<<", "<<line.left<<", "<<line.rght<<"]";
	return out;
}

inline std::ostream& operator<< (std::ostream& out, const LaneBlock &block)
{
	out<<"angle: "<<block.angle*180./CV_PI<<", ";
	out<<"count: "<<block.pointCnt <<", ";
	out<<"lines: "<<block.lineCnt<<", ";
	out<<"width: "<<(block.lineCnt ? block.pointCnt/block.lineCnt: 0)<<", ";
	out<<"length: "<<block.tear->row - block.head->row + 1<<", ";
	out<<"slope: "<<block.slope<<", ";
	out<<block.center_line;
	return out;
}


#endif//_LANE_DETECT_HPP_SJS_ZJY_

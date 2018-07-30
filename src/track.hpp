#ifndef _LANE_TRACK_HPP_SJS_ZJY_
#define _LANE_TRACK_HPP_SJS_ZJY_


#define M_PI_10             0.1745329
#define M_PI_05             0.0872665
#define M_PI_02             0.0349066

#define ROADLINE_MAX_ANGLE    M_PI_10  ///< 车道线在路面上的有效倾斜角度范围，即90度附近，左右最大倾斜10度
#define ROADLINE_VALID_ANGLE  M_PI_05  ///< 没有极远点约束时，左右最大倾斜5度

#define MAX_LANE_ROATE       (M_PI_02 * 1.5)
#define MIN_CANDILINE_ANGLE   M_PI_02
#define ASSIST_ONELINE        M_PI_02  ///< 角度差小于该值的，为同一条辅助线

#define MAX_ASSILINE_COUNT         50  ///< 辅助线的最大的计数，超过后不再增加计数
#define MAX_LANELINE_COUNT         50  ///< 候选车道线的计数，同上
#define VALID_ASSILINE_COUNT       10  ///< 辅助线超过该计数才做为有效，在补全车道线时使用
#define VALID_LANELINE_COUNT       15  ///< 连续出现该计数次数，则接受做为有效车道线
#define MAX_LANELINE_FAILED        20  ///< 候选车道线的连续失败最大计数
#define MAX_LANELINE_INTERVAL      50  ///< 候选车道线的计数区间，即上一次计数到现在
									   ///< 间隔的帧数超过此数，则计数清零，重新开始计数
#define MAX_TRACK_FAILED_COUNT     40  ///< 跟踪车道线失败次数超过该计数，则重新开始，可能是换到新道路上


#define DETECT_STEP_SLOW            3
#define DETECT_STEP_MEDIUM          2
#define DETECT_STEP_FAST            1

#define MAX_LINECOUNT              25
#define TRACKPARAM_ARR_LENGTH      10
#define VALID_CONFIRM_FRAME        10


/// 以下参数处理车道线路面投影
#define LANESHIFT_THRESH           40
#define MAX_LANEWIDTH_ERROR        50
#define LANE_WIDTH_DEFAULT        375   ///< 中国标准高速公路车道宽度：双向四车道是2*7.5米，
										///< 双向六车道是2*11.25米，双向八车道是2*15米。

//#define ADASLANE_HALF_IMAGE         1

/// 以下参数用来检测block
#if ADASLANE_HALF_IMAGE
//frames 2348, detected 2348 times, left line 2046, right line 2061
//VISUAL: use time: 1.515549 second, image count: 2348.
#define MAX_LANEBLOCK_WIDTH        26
#define MIN_LANEBLOCK_WIDTH        03
#define MIN_LANEBLOCK_LINECOUNT    12
#define MIN_LANEBLOCK_PIXELCOUNT  120
#define MIN_LANEBLOCK_DISTANCE     20
#else
//frames 2348, detected 2348 times, left line 2181, right line 2210
//VISUAL: use time: 5.240996 second, image count: 2348.
#define MAX_LANEBLOCK_WIDTH        54
#define MIN_LANEBLOCK_WIDTH        06
#define MIN_LANEBLOCK_LINECOUNT    24
#define MIN_LANEBLOCK_PIXELCOUNT  240
#define MIN_LANEBLOCK_DISTANCE     40
#endif

#define EDGE_GRADIENT_THRESH       10
#define MIN_LANEBLOCK_SLOPE     0.001 /// 0.035
#define MAX_LANEBLOCK_SLOPE     0.450 /// 0.450

// 调试选项
#if !ANDROID
//#define DEBUG_PRINT_MESS            1
//#define DEBUG_ALGORITHM             1
#endif

#include "core/stereo.hpp"
#include "detect.hpp"


struct TrackingLane
{
	int frame, succeed, failed, assiCount;
	float pointCnt, width, imageAngle, dist0, roadAngle;
	cv::Vec4f center_line;

	TrackingLane(): frame(-1), succeed(0), failed(0), assiCount(0),
			pointCnt(0), width(0), imageAngle(0), dist0(0), roadAngle(0)
	{}

	void CopyFrom(const LaneBlock *block)
	{
		pointCnt = block->pointCnt;
		width = block->pointCnt/block->lineCnt;
		imageAngle = block->angle;
		center_line = block->center_line;
	}
};


struct LaneLine
{
	int name, index, flag, solid, runin, confirmed;
	union {
		float distance;
		float angle;
	}prm;

	cv::Vec4f image_line, road_line;
	cv::Mat1i top, btm, mask;

	LaneLine(const int name): name(name), index(-20), flag(0), solid(-1), runin(0), confirmed(0),
			top(1, MAX_LINECOUNT), btm(1, MAX_LINECOUNT), mask(1, MAX_LINECOUNT)
	{
		prm.angle = 0;
		top = 3500; btm = 0;
	}

	void SetLaneLine(const LaneBlock &block, const int frameIndex);

	void SetLaneLine(const LaneBlock &block, const float dist0,
			const int frameIndex);

	void CheckLineFlag(int &flag, cv::Vec4f &line, const int frameIndex,
			const int delayFrames, const int minConfirm) const;

	void CheckLineFlag(int &flag, cv::Vec4f &line, const cv::Mat road2image,
			const int frameIndex, const int succeedIndex,
			const int delayFrames, const int minConfirm) const;

	void ToJson(const std::vector<cv::Vec3i> &vertexs, char jsonstr[1024]) const;

	void CopyTo(LaneLine &line) const
	{
		line.top = top;
		line.btm = btm;

		line.index = index;
		line.runin = runin;
		line.confirmed = confirmed;
	}

	void Reset()
	{
		top = cv::Mat1i(1, MAX_LINECOUNT);
		btm = cv::Mat1i(1, MAX_LINECOUNT);

		top = 3500; btm = 0;
		solid = -1; index = -20;
		runin = confirmed = 0;
		prm.distance = 0;
	}
};


struct BlockRoadParam
{
	double angle, x1500, x0000;
	cv::Vec4f roadline;
};

struct DpMatchLinesByFrm
{
	const int min_count, max_count, min_width, max_width;
	const float min_angle, max_angle, min_dist, max_dist;

	std::vector<short> candi_indexs, block_indexs;
	cv::Mat matches;

	DpMatchLinesByFrm(const int step): min_count(40/step), max_count(4000/step),
			min_width(5), max_width(40), min_angle(M_PI_10*.01),
			max_angle(M_PI_10), min_dist(10), max_dist(200)
	{}

	float Calc(const LaneBlock *const block, const TrackingLane *const cand,
			const float width);

	float Calc(const LaneBlock *const block, const TrackingLane *const cand,
			const float width, const BlockRoadParam *param);

	void operator()(const std::vector<LaneBlock*> &blocks,
			const std::vector<TrackingLane> &candlines, const uchar *flags);

	void operator()(const std::vector<LaneBlock*> &blocks,
			const std::vector<BlockRoadParam> &blockParams,
			const std::vector<TrackingLane> &candlines, const uchar *flags);
};


/**
 * 在LaneRecog对象得出的候选车道线基础上，将车道线投影到地面，
 * 根据车道宽度，车道线到地面坐标系原点的距离等信息，对候选车道线进行纠错，以及缺失时的补全处理
 */
class LaneTrack
{
public:

	enum LineType {
		NONE_LINE       = 0, // 0x0000,
		DETECTED_LINE   = 1, // 0x0001,
		CALCULATED_LINE = 2, // 0x0010,
		RUNIN_LINE      = 4, // 0x0100,
		CONFIRM_LINE    = 8,
	};

	int frameIndex, leftIndex , rghtIndex;
	LaneLine left, rght;

	const cv::Rect&rect;
	const std::vector<LaneBlock*> &blocks;

	static const char* GetVersion();

	virtual ~LaneTrack() {}

	LaneTrack(LaneBlockDetect *detect): frameIndex(0), leftIndex(-1), rghtIndex(-1),
			left(0), rght(1), rect(detect->GetDetectRect()), blocks(detect->GetLaneBlocks()),
			succeedCount(0), failedCount(0), detectCnt(0), succeedIndex(0), procesIndex(0),
			detectStep(1), lineCount(0), min_confirm(5), DpMatch(detect->GetRowStep()->stripe)
	{}

	void Forward() { frameIndex++; }

	bool HasUpdated() { return frameIndex == procesIndex; }

	bool NeedDetected() { return frameIndex - procesIndex >= detectStep; }

	float GetDetectStep() const { return detectStep; }

	int GetDetectCnt() { return detectCnt; }

	int GetLineCnt() { return lineCount; }

	uchar *GetFlags() { return flags.data(); }


	virtual void Reset()
	{
		frameIndex = 0; procesIndex = 0; detectCnt = 0;
		failedCount = 0; succeedCount = 0;
		left.Reset(); rght.Reset();
		trackobjs.clear();
	}

	void operator()(const cv::Mat gray, LaneBlockDetect *detect)
	{
		if( !NeedDetected() )
			return;
		cv::Sobel(gray, edgeImage, CV_16S, 1, 0, 3);
		(*detect)(edgeImage);
		operator()();
	}

	void operator()(const cv::Mat gray, LaneBlockDetect *detect,
			const cv::Size blockSize) // = cv::Size(4, 3)
	{
		if( !NeedDetected() )
			return;
		GenEdgeImage(gray, blockSize, edgeImage);
		(*detect)(edgeImage);
		operator()();
	}

	virtual void operator()();

	virtual int RecogLines();

	virtual void FrushStep(const int fast = DETECT_STEP_FAST,
			const int medium = DETECT_STEP_MEDIUM, const int slow = DETECT_STEP_SLOW);

	virtual void GetResLines(int &left_flag, cv::Vec4f &left_line,
			int &rght_flag, cv::Vec4f &rght_line, const int delay_frames = -1) const;

	virtual void UpdateTrackObjs();


	void CalcRectifyLine(const StereoRectify *const rectify,
			float lines[6], const int delay_frames = -1) const;

	bool CalcEndPoint(cv::Point2f &endpoint) const;

	bool CalcRectifyEndPoint(cv::Point2f &endpoint, const StereoRectify *const rectify) const;

	void EraseTrackObjs();

	void ConfirmLine(const float max_btm, const float valid_length);

	void RefrushTrackParam();

	virtual void Show(cv::Mat image);

protected:

	inline void CheckPassThrough();

	inline void _ConfirmLines(const float valid_length, const float max_btm);

	inline void _SignResLines(const float valid_length);

	/**
	 * 找到两条线时成功，succeedCount增加，failedCount归零，
	 * 一条也没找到时失败，failedCount增加，
	 * 连续失败MAX_TRACK_FAILED_COUNT次，succeedCount归零
	 */
	int succeedCount, failedCount, detectCnt;
	int succeedIndex, procesIndex, detectStep;
	int lineCount, lineCnts[3];

	const int min_confirm;

	cv::Mat1s edgeImage;
	std::vector<TrackingLane> trackobjs;
	std::vector<uchar> flags;
	DpMatchLinesByFrm DpMatch;
	cv::Point2f endpoints[TRACKPARAM_ARR_LENGTH], endpoint;
};

typedef class LaneTrack LaneTrackOpenModel;


class LaneTrackBasedRoad: public LaneTrack
{
public:

	virtual ~LaneTrackBasedRoad() {}

	LaneTrackBasedRoad(LaneBlockDetect *detect, const cv::Mat road2image, const cv::Mat image2road,
			const float car_width): LaneTrack(detect), road2image(road2image), image2road(image2road),
		alpha(.382), car_width(car_width), roadUsed(0), assiIndex(-1), lane_width(LANE_WIDTH_DEFAULT),
		lane_shift(0), lane_angle(0), diff_angle(0), diff_shift(0)
	{
		left.prm.distance = lane_shift - lane_width/2;
		rght.prm.distance = lane_shift + lane_width/2;

		memset(&lineCnts[0], 0, sizeof(int) * 3);
		memset(&laneWidths[0], 0, sizeof(double) * TRACKPARAM_ARR_LENGTH);
		memset(&laneShifts[0], 0, sizeof(double) * TRACKPARAM_ARR_LENGTH);
		memset(&endpoints[0], 0, sizeof(cv::Point2f) * TRACKPARAM_ARR_LENGTH);
	}

	float GetLaneWidth() const { return lane_width; }

	float GetLaneAngle() const { return lane_angle; }

	float GetLaneShift() const { return lane_shift; }


	virtual void Reset()
	{
		LaneTrack::Reset();
		lane_width = LANE_WIDTH_DEFAULT;
		lane_angle = lane_shift = 0;
		diff_angle = diff_shift = 0;
	}

	virtual void FrushStep(const int fast = DETECT_STEP_FAST,
			const int medium = DETECT_STEP_MEDIUM, const int slow = DETECT_STEP_SLOW);

	virtual void operator()();

	virtual int RecogLines();

	virtual void UpdateTrackObjs();


	virtual void GetResLines(int &left_flag, cv::Vec4f &left_line,
			int &rght_flag, cv::Vec4f &rght_line, const int delay_frames = -1) const;

	virtual void Show(cv::Mat image);

	void ShowRoadLines(cv::Mat &image, const float pixel_unit,
			const cv::Point shift = cv::Point());


private:

	inline void RecogAssit();

	inline void CheckLineRunin();

	inline void CheckPassThrough();

	inline void _ConfirmLines(const float valid_length, const float max_btm);

	inline void _SignResLines(const float valid_length);

	inline void UpdateParam();

	inline void CalcMissingLines();


protected:

	const cv::Mat road2image, image2road;
	const double alpha, car_width;

	int roadUsed, assiIndex;
	double lane_width, lane_shift, lane_angle, diff_angle, diff_shift;
	double laneWidths[TRACKPARAM_ARR_LENGTH], laneShifts[TRACKPARAM_ARR_LENGTH];

	std::vector<BlockRoadParam> blockParams;
};


void CalcRectifyLine(const StereoRectify *const rectify, const CameraParam &cam,
		cv::Vec4f line, float &k, float &b);

void TransLine(const cv::Vec4f src_line, cv::Vec4f &dst_line,
		const cv::Mat tran, const float step);

void FilterBlocks(const cv::Point2f endpoint, const std::vector<LaneBlock*> &blocks,
		uchar *flags, const float thresh = 10);

void CandSort(std::vector<TrackingLane> &vec);

inline void FormatLaneLine(LaneTrack *track, int index, char *prms)
{
	LaneLine &lline = index ? track->rght : track->left;
	int idx = index ? track->rghtIndex : track->leftIndex;
	lline.ToJson(track->blocks[idx]->vertexs, prms);
}

inline float CalcLineAngle(const cv::Vec4f road_line)
{
    float angle = atan2(road_line[0], road_line[1]);
	if( angle > CV_PI/2 )
		angle -= CV_PI;
	else if( angle < -CV_PI/2 )
		angle += CV_PI;
	return angle;
}


inline std::ostream& operator<< (std::ostream& out, const TrackingLane &line)
{
	out<<"image angle: "<<line.imageAngle*180./CV_PI<<", ";
	out<<"road angle: "<<line.roadAngle*180./CV_PI<<", ";
	out<<"dist0: "<<line.dist0<<", ";
	out<<"count: "<<line.pointCnt<<", ";
	out<<"width: "<<line.width<<", ";
	out<<"["<<line.succeed<<", "<<line.failed<<", "<<line.assiCount<<", "<<line.frame<<"]";

	return out;
}

inline std::ostream& operator<< (std::ostream& out, const LaneLine &line)
{
	out<<"["<<line.solid<<", ";
	out<<line.index<<", ";
	out<<line.confirmed<<", ";
	out<<line.runin<<", ";
	out<<line.image_line<<", ";
	out<<line.prm.distance<<"]";
	return out;
}

inline std::ostream& operator<< (std::ostream& out, const BlockRoadParam &param)
{
	out<<"[";//<<param.dist0<<", ";
	out<<param.x0000<<", ";
	out<<param.x1500<<", ";
	out<<param.angle*180./CV_PI<<", ";
	out<<param.roadline<<"]";
	return out;
}


#endif//_LANE_TRACK_HPP_SJS_ZJY_

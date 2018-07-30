#include "opencv2/highgui.hpp"

#define LOG_TAG "LaneDetect road.cpp"

#include "core/utils.hpp"
#include "track.hpp"


#ifndef M_PI_2
#define M_PI_2 (CV_PI/2)
#endif

static void Param2Line(cv::Vec4f &line, const float angle, const float delta)
{
    float _cos = cos(angle), _sin = sin(angle);
    cv::Matx33f rotate(_cos, _sin, 0, -_sin, _cos, 0, 0, 0, 1);
    std::vector<cv::Point2f> rts(3), pts(3);

    rts[0] = cv::Point2f(delta, 0);
    rts[1] = cv::Point2f(delta, 1000);
    rts[2] = cv::Point2f(delta, 2000);

    cv::perspectiveTransform(rts, pts, rotate);
    cv::fitLine(pts, line, cv::DIST_L2, 0, 0.1, 0.1);
}

cv::Vec4f GetParallelLine(const cv::Vec4f src, const float delta)
{
	const float sign = atan2(src[0], src[1]) < M_PI_2 ? 1 : -1;
	cv::Point2f c = cv::Point2f(src[1], src[0]) * delta * sign;
	return cv::Vec4f(src[0], src[1], src[2] +c.x, src[3] -c.y);
}

void LaneTrackBasedRoad::CalcMissingLines()
{
	if( leftIndex < 0 && rghtIndex > -1 )
	{
		left.road_line = GetParallelLine(rght.road_line, -lane_width);
	}
	else if( leftIndex > -1 && rghtIndex < 0 )
	{
		rght.road_line = GetParallelLine(left.road_line, lane_width);
	}
	else if( roadUsed )
	{
		Param2Line(left.road_line, lane_angle, lane_shift - lane_width/2);
		Param2Line(rght.road_line, lane_angle, lane_shift + lane_width/2);
	}
}

inline bool CheckLeftLine(const LaneBlock* block, const float x00)
{
	return block->angle > M_PI_2 || x00 < LANESHIFT_THRESH;
}

inline bool CheckRghtLine(const LaneBlock* block, const float x00)
{
	return block->angle < M_PI_2 || x00 > -LANESHIFT_THRESH;
}

int LaneTrackBasedRoad::RecogLines()
{
	const int blen = blocks.size();
	const int lhas = cv::sum(left.mask)[0], rhas = cv::sum(rght.mask)[0];
	int max_left_count = -1, max_rght_count = -1;
	int left_flags[blen], rght_flags[blen];
	float min_dist = MAX_LANEWIDTH_ERROR, lx15, rx15;
	cv::Vec4f rline;

	memset(left_flags, 0, sizeof(left_flags[0])*blen);
	memset(rght_flags, 0, sizeof(rght_flags[0])*blen);

	Param2Line(rline, lane_angle, lane_shift - lane_width/2);
	lx15 = rline[0]/rline[1] * (1500 - rline[3]) + rline[2];
	Param2Line(rline, lane_angle, lane_shift + lane_width/2);
	rx15 = rline[0]/rline[1] * (1500 - rline[3]) + rline[2];

	for( int i = blen-1; i >= 0; i-- )
	{
		const int cidx = DpMatch.block_indexs[i];
		if( cidx < 0 || trackobjs[cidx].assiCount >= VALID_ASSILINE_COUNT )
			continue;
		if( trackobjs[cidx].succeed <= VALID_LANELINE_COUNT/5 )
			continue;
		if( lhas < VALID_ASSILINE_COUNT && trackobjs[cidx].succeed < VALID_LANELINE_COUNT )
			continue;
		if( !CheckLeftLine(blocks[i], blockParams[i].x0000) )
			break;

		const int count = trackobjs[cidx].succeed - trackobjs[cidx].assiCount;
		if( max_left_count < count )
			max_left_count = count;
		left_flags[i] = 1;
	}

	for( int i = 0; i < blen; i++ )
	{
		const int cidx = DpMatch.block_indexs[i];
		if( cidx < 0 || trackobjs[cidx].assiCount >= VALID_ASSILINE_COUNT )
			continue;
		if( trackobjs[cidx].succeed <= VALID_LANELINE_COUNT/5 )
			continue;
		if( rhas < VALID_ASSILINE_COUNT && trackobjs[cidx].succeed < VALID_LANELINE_COUNT )
			continue;
		if( !CheckRghtLine(blocks[i], blockParams[i].x0000) )
			break;

		const int count = trackobjs[cidx].succeed - trackobjs[cidx].assiCount;
		if( max_rght_count < count )
		 	max_rght_count = count;
		rght_flags[i] = 1;
	}

	/// 1、在预测位置处有检测到的block，就选为车道线
	min_dist = MAX_LANEWIDTH_ERROR;
	for( int i = blen-1; i >= 0; i-- )
	{
		if( !left_flags[i] )
			continue;
	    float dist_pt0 = blockParams[i].x0000;// * blockParams[i].roadline[1];
		float dist_line = fabs(dist_pt0 - left.prm.distance)
				+ fabs(blockParams[i].x1500 - lx15);

		if( min_dist > dist_line )
		{
			min_dist = dist_line;
			leftIndex = i;
		}
	}

	min_dist = MAX_LANEWIDTH_ERROR;
	for( int i = 0; i < blen; i++ )
	{
		if( !rght_flags[i] )
			continue;
	    float dist_pt0 = blockParams[i].x0000;// * blockParams[i].roadline[1];
		float dist_line = fabs(dist_pt0 - rght.prm.distance) + fabs(blockParams[i].x1500 - rx15);

		if( min_dist > dist_line )
		{
			min_dist = dist_line;
			rghtIndex = i;
		}
	}

#if DEBUG_PRINT_MESS
	double l00 = 0, l15 = 0, r00 = 0, r15 = 0;
	if( leftIndex >= 0 )
	{
		l00 = blockParams[leftIndex].x0000;
		l15 = blockParams[leftIndex].x1500;
	}
	if( rghtIndex >= 0 )
	{
		r00 = blockParams[rghtIndex].x0000;
		r15 = blockParams[rghtIndex].x1500;
	}

	LOGI("expect left dist: [%f, %f], rght dist: [%f, %f]",
			left.prm.distance, lx15, rght.prm.distance, rx15);
	LOGI("actual left dist: [%f, %f], rght dist: [%f, %f]", l00, l15, r00, r15);
	LOGI("left index: %d, rght index: %d", leftIndex, rghtIndex);
#endif

	for( int i = 0; i < blen; i++ )
	{
		if( blocks[i]->lineCnt < MIN_LANEBLOCK_LINECOUNT
				|| blocks[i]->pointCnt < MIN_LANEBLOCK_PIXELCOUNT )
		{
			if( left_flags[i] )
				left_flags[i] = 0;
			if( rght_flags[i] )
				rght_flags[i] = 0;
		}
	}

	/// 2、选定的车道线内则紧贴着检测到另一条候选车道线，则选内则检测到的车道线
	if( leftIndex > -1 )
	{
		for( int j = leftIndex-1; j >= 0; j-- )
		{
			const int cidx = DpMatch.block_indexs[j];
			if( left_flags[j] == 0 || trackobjs[cidx].succeed < VALID_LANELINE_COUNT )
				continue;
			const int lidx = DpMatch.block_indexs[leftIndex];
			if( trackobjs[cidx].succeed*2 <= trackobjs[lidx].succeed )
				continue;

			if( !CheckLeftLine(blocks[j], blockParams[j].x0000) )
				break;

			float dd = blockParams[j].x0000 - blockParams[leftIndex].x0000;
			float dx = blockParams[j].x1500 - blockParams[leftIndex].x1500;
			if( dd < 10 || dd > 100 || dx < 10 || dx > 100 )
				break;
//			if( dd < -100 || dd > -10 || dx < 10 || dx > 100 )
//				break;

			leftIndex = j;
			break;
		}
	}

	if( rghtIndex > -1 )
	{
		for( int j = rghtIndex+1; j < blen; j++ )
		{
			const int cidx = DpMatch.block_indexs[j];
			if( rght_flags[j] == 0 || trackobjs[cidx].succeed < VALID_LANELINE_COUNT )
				continue;
			const int ridx = DpMatch.block_indexs[rghtIndex];
			if( trackobjs[cidx].succeed*2 <= trackobjs[ridx].succeed )
				continue;

			if( !CheckRghtLine(blocks[j], blockParams[j].x0000) )
				break;

			float dd = blockParams[j].x0000 - blockParams[rghtIndex].x0000;
			float dx = blockParams[j].x1500 - blockParams[rghtIndex].x1500;
			if( dd < -100 || dd > -10 || dx > -10 || dx < -100 )
				break;

			rghtIndex = j;
			break;
		}
	}

	/// 3、如果在预测的位置没有候选车道线，则重新选一条计数最大的候选线
	if( leftIndex < 0 && blen > 1 )
	{
		min_dist = LANE_WIDTH_DEFAULT;
		for( int j = blen-1; j >= 0; j-- )
		{
			if( left_flags[j] == 0 )
				continue;
			const int cidx = DpMatch.block_indexs[j], count = trackobjs[cidx].succeed;

			 if( max_left_count <= count*2 && count >= VALID_LANELINE_COUNT
					 && min_dist > blockParams[j].x0000 )
			 {
				 min_dist = blockParams[j].x0000;
				leftIndex = j;
			 }
		}
	}

	if( rghtIndex < 0 && blen > 1 )
	{
		min_dist = LANE_WIDTH_DEFAULT;
		for( int j = blen-1; j >= 0; j-- )
		{
			if( rght_flags[j] == 0 )
				continue;
			const int cidx = DpMatch.block_indexs[j], count = trackobjs[cidx].succeed;

			 if( max_rght_count <= count*2 && count >= VALID_LANELINE_COUNT
					 && min_dist > blockParams[j].x0000 )
			 {
				 min_dist = blockParams[j].x0000;
			 	rghtIndex = j;
			 }
		}
	}

	if( leftIndex >= 0 && leftIndex == rghtIndex )
	{
		if( blocks[leftIndex]->angle > M_PI_2 )
			rghtIndex = -1;
		else
			leftIndex = -1;
	}

	/// 4、如果选定的两条车道线不平行，或者车道宽度出现了突变，则必然是选错了
	if( leftIndex > -1 && rghtIndex > -1 && !image2road.empty() )
	{
		BlockRoadParam &lp = blockParams[leftIndex], &rp = blockParams[rghtIndex];

		if( fabs(lp.angle - rp.angle) > MAX_LANE_ROATE )
		{
			if( fabs(lp.angle) < fabs(rp.angle) )
				rghtIndex = -1;
			else
				leftIndex = -1;
		}

		if( fabs(rp.x0000 - lp.x0000 - LANE_WIDTH_DEFAULT) > 90 )
		{
			if( trackobjs[DpMatch.block_indexs[leftIndex]].succeed
					> trackobjs[DpMatch.block_indexs[rghtIndex]].succeed )
				rghtIndex = -1;
			else
				leftIndex = -1;
		}
	}

	if( leftIndex > -1 )
		lineCount++;
	if( rghtIndex > -1 )
		lineCount++;
	return lineCount;
}

void LaneTrackBasedRoad::RecogAssit()
{
	const int blen = blocks.size();
	float max_pixel = 0;

	for( int j = blen-1; j >= 0; j-- )
	{
		if( DpMatch.block_indexs[j] < 0 || j == leftIndex || j == rghtIndex )
			continue;

		if( leftIndex > -1 && blocks[j]->angle > M_PI_2 )
		{
			if( blocks[j]->angle < blocks[leftIndex]->angle )
				continue;

			float dd = blockParams[j].x0000 - blockParams[leftIndex].x0000;
			float dx = blockParams[j].x1500 - blockParams[leftIndex].x1500;

			if( dd > -10 || dd < -100 || dx > -10 || dx < -100 )
				continue;
		}

		if( rghtIndex > -1 && blocks[j]->angle < M_PI_2 )
		{
			if( blocks[j]->angle > blocks[rghtIndex]->angle )
				continue;

			float dd = blockParams[j].x0000 - blockParams[rghtIndex].x0000;
			float dx = blockParams[j].x1500 - blockParams[rghtIndex].x1500;

			if( dd > 100 || dd < 10 || dx < 10 || dx > 100 )
				continue;
		}

		if( max_pixel < blocks[j]->pointCnt )
		{
			max_pixel = blocks[j]->pointCnt;
			assiIndex = j;
		}
	}
}

void LaneTrackBasedRoad::UpdateTrackObjs()
{
	const int blen = blocks.size(), clen = trackobjs.size();
	int candIndex = assiIndex > -1? DpMatch.block_indexs[assiIndex] : -1;

	for( int j = 0; j < clen; j++ )
	{
		TrackingLane &line = trackobjs[j];
		if( DpMatch.candi_indexs[j] > -1
				&& frameIndex - line.frame < MAX_LANELINE_INTERVAL )
		{
			const int bidx = DpMatch.candi_indexs[j];

			if( line.succeed < MAX_LANELINE_COUNT )
				line.succeed++;
			line.failed= 0;
			line.frame = frameIndex;
			line.CopyFrom(blocks[bidx]);

			line.dist0 = blockParams[bidx].x0000;
			line.roadAngle = blockParams[bidx].angle;
		}
		else
			line.failed++;
	}

	for( int b = 0; b < blen; b++ )
	{
		short &cidx = DpMatch.block_indexs[b], hasNear = 0;
		if( !flags[b] || (cidx >= 0 &&
				frameIndex - trackobjs[cidx].frame < MAX_LANELINE_INTERVAL) )
			continue;

		for( int j = 0; j < clen; j++ )
		{
			TrackingLane &line = trackobjs[j];
			if( fabs(blocks[b]->angle - line.imageAngle) < MIN_CANDILINE_ANGLE )
			{
				hasNear = 1;
				break;
			}
		}

		if( hasNear == 0 )
		{
			TrackingLane line;

			line.frame = frameIndex;
			line.succeed = 1;
			line.failed = line.assiCount = 0;
			line.CopyFrom(blocks[b]);

			line.dist0 = blockParams[b].x0000;
			line.roadAngle = blockParams[b].angle;

			cidx = trackobjs.size();
			DpMatch.candi_indexs.push_back(b);
			trackobjs.push_back(line);
		}
	}

	if( assiIndex > -1 && candIndex > -1 )
	{
		float angle = blocks [assiIndex]->angle;
		TrackingLane &assist = trackobjs[candIndex];

		if( assist.assiCount && fabs(angle - assist.imageAngle) < ASSIST_ONELINE )
		{
			// 辅助线同侧必须存在已经确认的车道线，计数才可以增加
			// 否则计数不但不能增加，还要减少，以避免某次误判，将车道线判断成辅助线，从而始终无法纠正过来
			if( (assist.imageAngle > M_PI_2 && leftIndex > -1) ||
					(assist.imageAngle < M_PI_2 && rghtIndex > -1) )
			{
				if( assist.assiCount < MAX_ASSILINE_COUNT )
					assist.assiCount++;
			}
			else
				assist.assiCount--;
		}
		else if( (angle > M_PI_2 && leftIndex > -1) ||
				(angle < M_PI_2 && rghtIndex > -1) )
		{
			if( assist.assiCount < MAX_ASSILINE_COUNT )
				assist.assiCount++;
		}
		else
		{
			assiIndex = -1;
			candIndex = -1;
		}
	}

	for( int j = trackobjs.size()-1; j >= 0; j-- )
	{
		if( trackobjs[j].assiCount > 0 && j != candIndex )
			trackobjs[j].assiCount--;
	}
}

void LaneTrackBasedRoad::FrushStep(const int fast, const int medium, const int slow)
{
	if( frameIndex > procesIndex )
		return;
	const float shift = fabs(lane_shift);
	const float deviate = (lane_width - car_width) * .5;

	if( succeedCount || lineCount )
	{
		if( lineCount == 0 || shift*2 > deviate )
			detectStep = fast;
		else if( shift*4 < deviate )
			detectStep = MIN(MAX_LANELINE_INTERVAL/2, slow);
		else
			detectStep = MIN(MAX_LANELINE_INTERVAL/2, medium);
	}
	else
		detectStep = MIN(MAX_LANELINE_INTERVAL/2, slow);
}

/// 没有极远点约束时，车道线地面倾角最大为5度
static void FilterBlocks(const int bsize, const BlockRoadParam* params, uchar *flags)
{
	for( int i = bsize - 1; i >= 0; i-- )
	{
		flags[i] &= fabs(params[i].angle) < ROADLINE_VALID_ANGLE;
	}
}

void LaneTrackBasedRoad::operator()()
{
//	LOGI("%s: %s(%d).", __func__, __FILE__, __LINE__);
	const int blen = blocks.size();

	if( detectCnt % 200 == 0 )
		LOGI("lane lines have been detected %d times.", detectCnt);

	procesIndex = frameIndex; detectCnt++;
	lineCount = 0;
	leftIndex = rghtIndex = -1; assiIndex = -1;

#if DEBUG_PRINT_MESS
	LOGI("===== frame %d, detect %d blocks, %d candidate lines =====",
			frameIndex, blen, (int)trackobjs.size());
//#else
//	LOGI("===== frame %d, detect %d blocks, %d candidate lines =====",
//			frameIndex, blen, (int)trackobjs.size());
#endif

	if( blen )
	{
		blockParams.resize(blen);
		flags.resize(blen);
		memset(&flags[0],1, sizeof(flags[0])*blen);

		// 根据检测的候选车道线在路面上的投影，距离路面坐标原点的距离，投影线的走向进行筛选
		for( int i = 0; i < blen; i++ )
		{
			if( !flags[i] )
				continue;
			cv::Vec4f &cline = blockParams[i].roadline;
			TransLine(blocks[i]->center_line, cline, image2road, 20);

//			std::cout<<cline<<std::endl;

//			float k = cline[0]/cline[1];
//			blockParams[i].x0000 = k * (0 - cline[3]) + cline[2];
//			blockParams[i].x1500 = k * (1500 - cline[3]) + cline[2];

			blockParams[i].x0000 = cline[0] * (0 - cline[3]) + cline[2]*cline[1];
			blockParams[i].x1500 = cline[0] * (1500 - cline[3]) + cline[2]*cline[1];
			blockParams[i].angle = CalcLineAngle(cline);

			flags[i] = fabs(blockParams[i].x0000 - left.prm.distance) < MAX_LANEWIDTH_ERROR*2 ||
					fabs(blockParams[i].x0000 - rght.prm.distance) < MAX_LANEWIDTH_ERROR*2;
			// 车道线最大倾斜角为10度
			flags[i] &= fabs(blockParams[i].angle) < ROADLINE_MAX_ANGLE;

			if( succeedCount > VALID_LANELINE_COUNT )
				flags[i] &= fabs(blockParams[i].angle - lane_angle) < MAX_LANE_ROATE;
		}

		if( endpoint != cv::Point2f(0, 0) )
			FilterBlocks(endpoint, blocks, &flags[0], 50);
		else
			FilterBlocks(blocks.size(), blockParams.data(), &flags[0]);

		DpMatch(blocks, blockParams, trackobjs, &flags[0]);
		RecogLines();
		CheckPassThrough();
		RecogAssit();

#if DEBUG_PRINT_MESS
		for( int i = trackobjs.size()-1; i >= 0; i-- )
		{
			std::stringstream candline_outstr;
			candline_outstr<<trackobjs[i];
			LOGI("candlines[%d]: %s", i, candline_outstr.str().c_str());
		}

		for( int i = blen - 1; i >= 0; i-- )
		{
			std::stringstream block_outstr, param_outstr;

			block_outstr<<*blocks[i];
			LOGI("blocks[%d]: %s", i, block_outstr.str().c_str());

			param_outstr<<blockParams[i];
			LOGI("params[%d]: %s", i, param_outstr.str().c_str());
		}

		std::stringstream flags_outstr;
		cv::Mat iflags;
		cv::Mat(flags).convertTo(iflags, CV_32S);
		flags_outstr<<iflags.reshape(1, 1);
		LOGI("flags: %s", flags_outstr.str().c_str());

		std::stringstream indexs_outstr;
		indexs_outstr<<DpMatch.candi_indexs;
		LOGI("candi_index: %s", indexs_outstr.str().c_str());

		indexs_outstr.clear(); indexs_outstr.str("");
		indexs_outstr<<DpMatch.block_indexs;
		LOGI("block_index: %s", indexs_outstr.str().c_str());

		LOGI("assist index: %d, left index: %d, rght index: %d",
				assiIndex, leftIndex, rghtIndex);
		LOGI("lane angle: %f, shift: %f, width: %f, diff angle: %f, shift: %f",
				lane_angle*180/M_PI, lane_shift, lane_width,
				diff_angle*180/M_PI, diff_shift);
#endif

		UpdateParam();
		left.road_line = leftIndex > -1 ?
				blockParams[leftIndex].roadline : cv::Vec4f(0, 0, 0, 0);
		rght.road_line = rghtIndex > -1 ?
				blockParams[rghtIndex].roadline : cv::Vec4f(0, 0, 0, 0);

		if( lineCount < 2 )
			CalcMissingLines();
		UpdateTrackObjs();
		CandSort(trackobjs);
	}
	else
	{
		for( int i = trackobjs.size()-1; i >= 0; i-- )
			trackobjs[i].failed++;
	}

	float valid_length = 1000, max_btm = 1500;
	_ConfirmLines(valid_length, max_btm);
	_SignResLines(valid_length);
	ConfirmLine(max_btm, valid_length);

	CheckLineRunin();
	EraseTrackObjs();
	RefrushTrackParam();

	//	cv::Vec4f _left_line(0, 0, 0, 0), _rght_Line(0, 0, 0, 0);
	left.flag = rght.flag = NONE_LINE;
	left.CheckLineFlag(left.flag, left.image_line, road2image,
			frameIndex, succeedIndex, detectStep*2, min_confirm);
	rght.CheckLineFlag(rght.flag, rght.image_line, road2image,
			frameIndex, succeedIndex, detectStep*2, min_confirm);
//	left.CheckLineFlag(left.flag, left.image_line, frameIndex, detectStep*2, min_confirm);
//	rght.CheckLineFlag(rght.flag, rght.image_line, frameIndex, detectStep*2, min_confirm);

#if DEBUG_PRINT_MESS
	LOGI("succeed index: %d, count: %d, failed count: %d, detect step: %d, shift: %f.",
			succeedIndex, succeedCount, failedCount, detectStep, lane_shift);

	std::stringstream linecnt_outstr;
	linecnt_outstr<<cv::Mat(1, 3, CV_32SC1, lineCnts);
	LOGI("line count: %s", linecnt_outstr.str().c_str());

	std::stringstream lines_str;
	lines_str<<"left: "<<left<<std::endl;
	lines_str<<"rght: "<<rght;
	LOGI("%s", lines_str.str().c_str());
#endif
}

/// 防车道线抖动
void LaneTrackBasedRoad::CheckPassThrough()
{
	float thres = (lane_width + LANESHIFT_THRESH)/2;

	if( lane_shift > thres )
	{
		LOGI("copy left to rght.");
		left.CopyTo(rght);
		left.Reset();
		lane_shift -= lane_width;

		if( rghtIndex > -1 )
			lineCount--;
		rghtIndex = leftIndex;
		leftIndex = -1;
	}
	else if( lane_shift < -thres )
	{
		LOGI("copy rght to left.");
		rght.CopyTo(left);
		rght.Reset();
		lane_shift += lane_width;

		if( leftIndex > -1 )
			lineCount--;
		leftIndex = rghtIndex;
		rghtIndex = -1;
	}
}

void LaneTrackBasedRoad::UpdateParam()
{
	const int frames = frameIndex - succeedIndex, arrIdx = detectCnt%TRACKPARAM_ARR_LENGTH;
	float angle = 0, shift = 0, width = 0, width2 = lane_width/2;

	if( lineCount == 2 )
	{
	    float left_dist = blockParams[leftIndex].x0000;
	    float rght_dist = blockParams[rghtIndex].x0000;

	    angle = (blockParams[leftIndex].angle + blockParams[rghtIndex].angle)*.5;
	    width = rght_dist - left_dist;
	    shift = (rght_dist + left_dist)*.5;
	}
	else if( leftIndex > -1 )
	{
	    float left_dist = blockParams[leftIndex].x0000;
	    angle = blockParams[leftIndex].angle;
	    shift = left_dist + width2;
	}
	else if( rghtIndex > -1 )
	{
	    float rght_dist = blockParams[rghtIndex].x0000;
	    angle = blockParams[rghtIndex].angle;
	    shift = rght_dist - width2;
	}

	laneWidths[arrIdx] = width;
	laneShifts[arrIdx] = shift;

	if( lineCount == 2 )
	{
		if( lineCnts[2] >= VALID_LANELINE_COUNT && roadUsed == 0 )
		{
			float total = 0, count = 0;

			for( int i = 0; i < TRACKPARAM_ARR_LENGTH; i++ )
			{
				if( laneWidths[i] > 200 )
				{
					count ++;
					total += laneWidths[i];
				}
			}

			if( count > 0 )
			{
				lane_width = total/count;
				roadUsed = 1;
			}
		}
		else if( lineCnts[2] >= VALID_LANELINE_COUNT/2 )
			lane_width = lane_width * (1-alpha) + width * alpha;
	}
	else if( lineCnts[2] < VALID_LANELINE_COUNT/3 && roadUsed )
		roadUsed = 0;

	if( lineCount > 0 )
	{
		const float _alpha = alpha / frames;
		float da = angle - lane_angle, ds = shift - lane_shift;

		if( ds > width2 )
			ds -= lane_width;
		else if( ds < -width2 )
			ds += lane_width;

		diff_angle = diff_angle * (1-alpha) + da * _alpha;
		diff_shift = diff_shift * (1-alpha) + ds * _alpha;
		lane_angle = angle;
		lane_shift = shift;
//		lane_angle = lane_angle * (1-alpha) + angle * alpha;
//		lane_shift = lane_shift * (1-alpha) + shift * alpha;
	}
	else if( frames < detectStep*4 )
	{
		lane_angle += diff_angle;
		lane_shift += diff_shift;
	}
}

static void TranVertex(std::vector<cv::Point2f> &rps, const LaneBlock *block,
		cv::Mat image2road, const float deltay)
{
	const EdgeLine *head = block->head, *tear = block->tear;
	std::vector<cv::Point2f> ips(2);

	ips[0] = cv::Point2f((head->left+head->rght)/2, head->row + deltay);
	ips[1] = cv::Point2f((tear->left+tear->rght)/2, tear->row + deltay);

	cv::perspectiveTransform(ips, rps, image2road);
}

void LaneTrackBasedRoad::_SignResLines(const float valid_length)
{
	std::vector<cv::Point2f> rps(2);

	if( leftIndex > -1 )
	{
	    float left_dist = blockParams[leftIndex].x0000;// * blockParams[leftIndex].roadline[1];
		left.SetLaneLine(*blocks[leftIndex], left_dist, frameIndex);
		left.mask.at<int>(detectCnt % MAX_LINECOUNT) = 1;

		if( blocks[leftIndex]->vertexs.size() == 2 )
		{
			TranVertex(rps, blocks[leftIndex], image2road, rect.y);
			if( fabs(rps[0].y - rps[1].y) > valid_length )
				left.solid = 1;
		}
//		LOGI("1, left dist: %f", left.prm.distance);
	}
	else
	{
//		left.prm.distance = lane_width/2 - lane_shift;
		left.prm.distance = lane_shift - lane_width/2;
		left.solid = -1;
		left.mask.at<int>(detectCnt % MAX_LINECOUNT) = 0;
//		LOGI("2, left dist: %f", left.prm.distance);
	}

	if( rghtIndex > -1 )
	{
	    float rght_dist = blockParams[rghtIndex].x0000;// * blockParams[rghtIndex].roadline[1];
		rght.SetLaneLine(*blocks[rghtIndex], rght_dist, frameIndex);
		rght.mask.at<int>(detectCnt % MAX_LINECOUNT) = 1;

		if( blocks[rghtIndex]->vertexs.size() == 2 )
		{
			TranVertex(rps, blocks[rghtIndex], image2road, rect.y);
			if( fabs(rps[0].y - rps[1].y) > valid_length )
				rght.solid = 1;
		}
	}
	else
	{
		rght.prm.distance = lane_shift + lane_width/2;
		rght.solid = -1;
		rght.mask.at<int>(detectCnt % MAX_LINECOUNT) = 0;
	}

//	LOGI("2, expect left dist: %f, rght dist: %f",
//			left.prm.distance, rght.prm.distance);
}

void LaneTrackBasedRoad::_ConfirmLines(const float valid_length, const float max_btm)
{
	const int confirm_idx = detectCnt % MAX_LINECOUNT;
	std::vector<cv::Point2f> rps(2);

	if( leftIndex > -1 )
	{
		TranVertex(rps, blocks[leftIndex], image2road, rect.y);
		left.top.at<int>(confirm_idx) = rps[1].y;
		left.btm.at<int>(confirm_idx) = rps[0].y;
	}
	else
	{
		left.top.at<int>(confirm_idx) = max_btm;
		left.btm.at<int>(confirm_idx) = 0;
	}

	if( rghtIndex > -1 )
	{
		TranVertex(rps, blocks[rghtIndex], image2road, rect.y);
		rght.top.at<int>(confirm_idx) = rps[1].y;
		rght.btm.at<int>(confirm_idx) = rps[0].y;
	}
	else
	{
		rght.top.at<int>(confirm_idx) = max_btm;
		rght.btm.at<int>(confirm_idx) = 0;
	}
}

void LaneTrackBasedRoad::CheckLineRunin()
{
	const float deviate1 = (lane_width - car_width)/2;
	left.runin = rght.runin = 0;

	for( int i = 0; i < TRACKPARAM_ARR_LENGTH; i++ )
	{
		if( laneShifts[i] > deviate1 )
			left.runin++;
		if( laneShifts[i] <-deviate1 )
			rght.runin++;
	}
}

void LaneLine::CheckLineFlag(int &flag, cv::Vec4f &line, const cv::Mat road2image,
		const int frameIndex, const int succeedIndex, const int delayFrames, const int minConfirm) const
{
	if( frameIndex - index < delayFrames )
	{
		flag = LaneTrack::DETECTED_LINE;
		line = image_line;
	}
	else if( fabs(road_line[1]) > .01
			&& frameIndex - succeedIndex < VALID_CONFIRM_FRAME*2 )
	{
		flag = LaneTrack::CALCULATED_LINE;
		TransLine(road_line, line, road2image, 1000);
	}

	if( frameIndex - index < VALID_CONFIRM_FRAME )
	{
		if( confirmed > minConfirm )
			flag += LaneTrack::CONFIRM_LINE;
		if( runin >= 5 )
			flag += LaneTrack::RUNIN_LINE;
	}
}

void LaneTrackBasedRoad::GetResLines(int &_left_flag, cv::Vec4f &_left_line,
		int &_rght_flag, cv::Vec4f &_rght_line, int delay_frames) const
{
	if( delay_frames <= 0 )
		delay_frames = detectStep;
	_left_flag = _rght_flag = NONE_LINE;

	left.CheckLineFlag(_left_flag, _left_line, road2image,
			frameIndex, succeedIndex, delay_frames, min_confirm);
	rght.CheckLineFlag(_rght_flag, _rght_line, road2image,
			frameIndex, succeedIndex, delay_frames, min_confirm);
}

void LaneTrackBasedRoad::Show(cv::Mat image)
{
	const cv::Scalar yellow(0, 255, 255);
	const int linewidth = 2;

	LaneTrack::Show (image);
	if( assiIndex > -1 )
	{
		LaneBlock* block = blocks[assiIndex];
		DrawDottedLine(image, block->center_line, yellow, linewidth);
	}
}

inline void DrawBirdLine(cv::Mat &bird, cv::Vec4f line,
		const cv::Scalar color, const float pixel_unit, const cv::Point shift)
{
	cv::Point bpt = road2birdview(cv::Point(line[2], line[3]),
					bird.size(), pixel_unit, shift);
	line[1] = -line[1]; line[2] = bpt.x; line[3] = bpt.y;
	DrawSolidLine(bird, line, color, 1);
}

inline void DrawBirdPts(cv::Mat &bird, const std::vector<cv::Point2f> &rps,
		const cv::Scalar color, const float pixel_unit, const cv::Point shift)
{
	const int len = rps.size();
	std::vector<cv::Point2f> bps(len);

	road2birdview(rps, bps, bird.size(), pixel_unit, shift);
	for( int i = 0; i < len; i++ )
		cv::circle(bird, bps[i], 3, color);
}

void LaneTrackBasedRoad::ShowRoadLines(cv::Mat &bdimg, const float pixel_unit,
		const cv::Point shift)
{
	if( pixel_unit < 0.1 )
		return;
	const cv::Scalar blue(255, 0, 0), green(0, 255, 0),
			yellow(0, 255, 255), red(0, 0, 255);
	std::vector<cv::Point2f> rps(20), bps(20);

	for( int i = 0; i < 10; i++ )
	{
		rps[2*i+0] = cv::Point2f(-400, i*500);
		rps[2*i+1] = cv::Point2f(+400, i*500);
	}

	road2birdview(rps, bps, bdimg.size(), pixel_unit, shift);
	for( int i = 0; i < 10; i++ )
		cv::line(bdimg, bps[2*i+0], bps[2*i+1], blue);

	DrawBirdLine(bdimg, left.road_line, green, pixel_unit, shift);
	DrawBirdLine(bdimg, rght.road_line, green, pixel_unit, shift);
}

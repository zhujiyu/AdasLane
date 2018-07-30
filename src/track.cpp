#include "opencv2/highgui.hpp"

#define LOG_TAG "LaneDetect track.cpp"

#include "core/utils.hpp"
#include "track.hpp"

#ifndef M_PI_2
#define M_PI_2      (CV_PI/2)
#endif

#ifndef M_PI_4
#define M_PI_4      (CV_PI/4)
#endif

static const char* version = "lane 2.5.0417 @2015-2018 Shuangjisha Inc.\n";
const char* LaneTrack::GetVersion() { return version; }

#define MAX_VEC(x, y, z) MAX( MAX( (x), (y) ), (z) )

float DpMatchLinesByFrm::Calc(const LaneBlock *const block,
		const TrackingLane *const cand, const float width)
{
	float dcount = std::abs(block->pointCnt - cand->pointCnt);
	float dwidth = fabs(width - cand->width);
	float dangle = fabs(block->angle - cand->imageAngle);
	float sangle = 0, scount = 0, swidth = 0;

	if( dangle < min_angle )
		sangle = 1;
	else if( dangle < max_angle )
		sangle = min_angle / dangle;

	if( dcount < min_count )
		scount = 1;
	else if( dcount < max_count )
		scount = min_count / dcount;

	if( dwidth < min_width )
		swidth = 1;
	else if( dwidth < max_width )
		swidth = min_width / dwidth;

	return sangle * scount * swidth;
}

float DpMatchLinesByFrm::Calc(const LaneBlock *const block, const TrackingLane *const cand,
		const float width, const BlockRoadParam *param)
{
	float dcount = std::abs(block->pointCnt - cand->pointCnt);
	float dwidth = fabs(width - cand->width);
	float dangle = fabs(block->angle - cand->imageAngle);
	float sangle = 0, scount = 0, swidth = 0;

	if( dangle < min_angle )
		sangle = 1;
	else if( dangle < max_angle )
		sangle = min_angle / dangle;

	if( dcount < min_count )
		scount = 1;
	else if( dcount < max_count )
		scount = min_count / dcount;

	if( dwidth < min_width )
		swidth = 1;
	else if( dwidth < max_width )
		swidth = min_width / dwidth;

	float ddist0 = fabs(param->x0000 - cand->dist0);
	float droada = fabs(param->angle - cand->roadAngle);
	float sdist0 = 0, srangle = 0;

	if( ddist0 < min_dist )
		sdist0 = 1;
	else if( ddist0 < max_dist )
		sdist0 = min_dist / ddist0;

	if( droada < min_angle )
		srangle = 1;
	else if( droada < max_angle )
		srangle = min_angle / droada;

	return sangle * scount * swidth * sdist0 * srangle;
//	return sangle * scount * swidth * sdist0 * srangle*
//			(cand->succeed/float(MAX_ASSILINE_COUNT));
}

void DpMatchLinesByFrm::operator()(const std::vector<LaneBlock*> &blocks,
		const std::vector<TrackingLane> &candlines, const uchar *flags)
{
	const int blen = blocks.size(), clen = candlines.size();
	int bidx = 0, cidx = 0;
	float maxscore = 0;

	candi_indexs.resize(clen);
	block_indexs.resize(blen);

	memset(block_indexs.data(), -1, sizeof(short)*blen);
	memset(candi_indexs.data(), -1, sizeof(short)*clen);

	matches.create(blen+1, clen+1, CV_32FC1);
	matches = 0;

	for( int b = 0; b < blen; b++ )
	{
		float *cmatch = matches.ptr<float>(b+1);
		float *pmatch = matches.ptr<float>(b);
		float width = blocks[b]->pointCnt/blocks[b]->lineCnt;

		if( flags && !flags[b] )
		{
			memcpy(cmatch+1, pmatch+1, clen*sizeof(float));
			continue;
		}

		for( int c = 0; c < clen; c++ )
		{
			float score = Calc(blocks[b], &candlines[c], width);
			cmatch[c+1] = MAX_VEC(pmatch[c+1], cmatch[c], pmatch[c] + score);
//			scores.at<float>(b, c) = score;

			if( maxscore < cmatch[c+1] )
			{
				bidx = b; cidx = c;
				maxscore = cmatch[c+1];
			}
		}
	}
//	std::cout<<scores<<std::endl;
//	std::cout<<matches<<std::endl;

	while( maxscore > 0 )
	{
//		LOGI("max score: %f, row: %d, col: %d", maxscore, bidx, cidx);
		block_indexs[bidx] = cidx; candi_indexs[cidx] = bidx;
		maxscore = 0;
		int btm = bidx, rgt = cidx;

		for( int b = 0; b < btm; b++ )
		{
			float *cmatch = matches.ptr<float>(b+1);

			for( int c = 0; c < rgt; c++ )
			{
				if( maxscore < cmatch[c+1] )
				{
					bidx = b; cidx = c;
					maxscore = cmatch[c+1];
				}
			}
		}
	}
}

void DpMatchLinesByFrm::operator()(const std::vector<LaneBlock*> &blocks,
		const std::vector<BlockRoadParam> &blockParams,
		const std::vector<TrackingLane> &candlines, const uchar *flags)
{
	const int blen = blocks.size(), clen = candlines.size();
	int bidx = 0, cidx = 0;
	float maxscore = 0;

	candi_indexs.resize(clen);
	block_indexs.resize(blen);

	memset(block_indexs.data(), -1, sizeof(short)*blen);
	memset(candi_indexs.data(), -1, sizeof(short)*clen);

	matches.create(blen+1, clen+1, CV_32FC1);
	matches = 0;

	for( int b = 0; b < blen; b++ )
	{
		float *cmatch = matches.ptr<float>(b+1);
		float *pmatch = matches.ptr<float>(b);
		float width = blocks[b]->pointCnt/blocks[b]->lineCnt;

		if( flags && !flags[b] )
		{
			memcpy(cmatch+1, pmatch+1, clen*sizeof(float));
			continue;
		}

		for( int c = 0; c < clen; c++ )
		{
			float score = Calc(blocks[b], &candlines[c], width, &blockParams[b]);
			cmatch[c+1] = MAX_VEC(pmatch[c+1], cmatch[c], pmatch[c] + score);
//			scores.at<float>(b, c) = score;

			if( maxscore < cmatch[c+1] )
			{
				bidx = b; cidx = c;
				maxscore = cmatch[c+1];
			}
		}
	}
//	std::cout<<scores<<std::endl;
//	std::cout<<matches<<std::endl;

	while( maxscore > 0 )
	{
//		LOGI("max score: %f, row: %d, col: %d", maxscore, bidx, cidx);
		block_indexs[bidx] = cidx; candi_indexs[cidx] = bidx;
		maxscore = 0;
		int btm = bidx, rgt = cidx;

		for( int b = 0; b < btm; b++ )
		{
			float *cmatch = matches.ptr<float>(b+1);

			for( int c = 0; c < rgt; c++ )
			{
				if( maxscore < cmatch[c+1] )
				{
					bidx = b; cidx = c;
					maxscore = cmatch[c+1];
				}
			}
		}
	}
}


void LaneLine::ToJson(const std::vector<cv::Vec3i> &vertexs, char jsonstr[1024]) const
{
	const int len = vertexs.size();
	const char* fmt = "{"
			"\"continuity\": %d,"
			"\"detected_frame_index\": %d,"
			"\"runin\": %d,"
			"\"confirmed\": %d,"

			"\"image_line\": [%f,%f,%f,%f],"
			"\"road_line\": [%f,%f,%f,%f],"
			"\"distance\": %f,"
			"\"vertexs\": [%s]"
			" }";
	char verstr[1024];

	memset(jsonstr, 0, sizeof(char)*1024);
	memset(verstr , 0, sizeof(char)*1024);

	if( len > 0 && len < 10 && len%2 == 0 )
	{
		sprintf(verstr, "{\"row\": %d, \"left\": %d, \"right\": %d}",
				vertexs[0][0], vertexs[0][1], vertexs[0][2]);
		for( int i = 1; i < len; i++ )
		{
			sprintf(verstr, "%s, {\"row\": %d, \"left\": %d, \"right\": %d}",
					verstr, vertexs[i][0], vertexs[i][1], vertexs[i][2]);
		}
	}

	sprintf(jsonstr, fmt, solid, index, runin, confirmed,
			image_line[0], image_line[1], image_line[2], image_line[3],
			road_line[0], road_line[1], road_line[2], road_line[3],
			prm.angle, verstr);
}

void LaneLine::SetLaneLine(const LaneBlock &block, const int frameIndex)
{
	solid = 0;
	index = frameIndex;
	prm.angle = block.angle;
	image_line = block.center_line;
}

void LaneLine::SetLaneLine(const LaneBlock &block, const float dist0,
		const int frameIndex)
{
	solid = 0;
	index = frameIndex;
	prm.distance = dist0;
	image_line = block.center_line;
}

void LaneLine::CheckLineFlag(int &flag, cv::Vec4f &line, const int frameIndex,
		const int delayFrames, const int minConfirm) const
{
	if( frameIndex - index < delayFrames )
	{
		flag = LaneTrack::DETECTED_LINE;
		line = image_line;
	}

	if( frameIndex - index < VALID_CONFIRM_FRAME )
	{
		if( confirmed > minConfirm )
			flag += LaneTrack::CONFIRM_LINE;
		if( runin >= 5 )
			flag += LaneTrack::RUNIN_LINE;
	}
}


bool LaneTrack::CalcEndPoint(cv::Point2f &endpoint) const
{
	endpoint = cv::Point2f(0, 0);
	if( lineCount == 2 )
	{
		return InterPt(endpoint,
				((LaneBlock*)blocks[leftIndex])->center_line,
				((LaneBlock*)blocks[rghtIndex])->center_line);
	}
	return false;
}

bool LaneTrack::CalcRectifyEndPoint(cv::Point2f &endpoint, const StereoRectify *const rectify) const
{
	cv::Point2f oept(0, 0);
	bool res = CalcEndPoint(oept);
	if( res )
		rectify->toRectifyPoint(oept, endpoint, &rectify->camera->cam_left);
	return res;
}

void FilterBlocks(const cv::Point2f endpoint, const std::vector<LaneBlock*> &blocks,
		uchar *flags, const float thresh)
{
	for( int i = blocks.size()-1; i >= 0; i-- )
	{
		if( flags[i] )
		{
			cv::Vec4f line = ((LaneBlock*)blocks[i])->center_line;
			float d = point2line(line.val, endpoint.x, endpoint.y);
			flags[i] &= d < thresh ? 1 :0;
		}
	}
}

int LaneTrack::RecogLines()
{
	const int blen = blocks.size(), lhas = cv::sum(left.mask)[0], rhas = cv::sum(rght.mask)[0];
	short left_flags[blen], rght_flags[blen], max_left_count = -1, max_rght_count = -1;
	float min_angle = M_PI_05;
	cv::Vec4f rline;

	memset(left_flags, 0, sizeof(left_flags[0])*blen);
	memset(rght_flags, 0, sizeof(rght_flags[0])*blen);

	for( int i = blen-1; i >= 0 && blocks[i]->angle > M_PI_2; i-- )
	{
		const int cidx = DpMatch.block_indexs[i];

		if( cidx < 0 || trackobjs[cidx].assiCount >= VALID_ASSILINE_COUNT )
			continue;
		if( trackobjs[cidx].succeed <= VALID_LANELINE_COUNT/5 )
			continue;
		if( lhas < VALID_ASSILINE_COUNT && trackobjs[cidx].succeed < VALID_LANELINE_COUNT )
			continue;

		const int count = trackobjs[cidx].succeed - trackobjs[cidx].assiCount;

		if( max_left_count < count )
			max_left_count = count;
		left_flags[i] = 1;
	}

	for( int i = 0; i < blen && blocks[i]->angle < M_PI_2; i++ )
	{
		const int cidx = DpMatch.block_indexs[i];

		if( cidx < 0 || trackobjs[cidx].assiCount >= VALID_ASSILINE_COUNT )
			continue;
		if( trackobjs[cidx].succeed <= VALID_LANELINE_COUNT/5 )
			continue;
		if( rhas < VALID_ASSILINE_COUNT && trackobjs[cidx].succeed < VALID_LANELINE_COUNT )
			continue;

		const int count = trackobjs[cidx].succeed - trackobjs[cidx].assiCount;

		if( max_rght_count < count )
		 	max_rght_count = count;
		rght_flags[i] = 1;
	}

	/// 1、在预测位置处有检测到的block，就选为车道线
	min_angle = M_PI_05;
	for( int i = blen-1; i >= 0; i-- )
	{
		if( !left_flags[i] )
			continue;
		float da = fabs(blocks[i]->angle - left.prm.angle);

		if( da < min_angle )
		{
			min_angle = da;
			leftIndex = i;
		}
	}

	min_angle = M_PI_05;
	for( int i = 0; i < blen; i++ )
	{
		if( !rght_flags[i] )
			continue;
		float da = fabs(blocks[i]->angle - rght.prm.angle);

		if( da < min_angle )
		{
			min_angle = da;
			rghtIndex = i;
		}
	}

#if DEBUG_PRINT_MESS
	LOGI("expect left angle: %f, rght angle: %f", left.prm.angle, rght.prm.angle);
	LOGI("actual left angle: %f, rght agnle: %f", leftIndex >= 0? blocks[leftIndex]->angle : 0,
			rghtIndex >= 0? blocks[rghtIndex]->angle : 0);
	LOGI("left index: %d, rght index: %d", leftIndex, rghtIndex);
#endif

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

			if( blocks[j]->angle > M_PI_2 && blocks[j]->lineCnt > MIN_LANEBLOCK_LINECOUNT
					&& blocks[j]->pointCnt > MIN_LANEBLOCK_PIXELCOUNT )
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

			if( blocks[j]->angle < M_PI_2 && blocks[j]->lineCnt > MIN_LANEBLOCK_LINECOUNT
					&& blocks[j]->pointCnt > MIN_LANEBLOCK_PIXELCOUNT )
				rghtIndex = j;
			break;
		}
	}
//	LOGI("left index: %d, rght index: %d", leftIndex, rghtIndex);

	/// 3、如果在预测的位置没有候选车道线，则重新选一条计数最大的候选线
	if( leftIndex < 0 && blen > 1 )
	{
		min_angle = LANE_WIDTH_DEFAULT;
		for( int j = blen-1; j >= 0; j-- )
		{
			if( left_flags[j] == 0 )
				continue;
			const int cidx = DpMatch.block_indexs[j], count = trackobjs[cidx].succeed;

			if( max_left_count <= count*2 && count >= VALID_LANELINE_COUNT
					 && min_angle > blocks[j]->angle )
			{
				min_angle = blocks[j]->angle;
				leftIndex = j;
			}
		}
	}

	if( rghtIndex < 0 && blen > 1 )
	{
		min_angle = LANE_WIDTH_DEFAULT;
		for( int j = 0; j < blen; j++ )
		{
			if( rght_flags[j] == 0 )
				continue;
			const int cidx = DpMatch.block_indexs[j], count = trackobjs[cidx].succeed;

			if( max_rght_count <= count*2 && count >= VALID_LANELINE_COUNT
					 && min_angle > blocks[j]->angle )
			{
				min_angle = blocks[j]->angle;
			 	rghtIndex = j;
			}
		}
	}
//	LOGI("left index: %d, rght index: %d", leftIndex, rghtIndex);

	if( leftIndex > -1 )
		lineCount++;
	if( rghtIndex > -1 )
		lineCount++;
	return lineCount;
}


///// 处理候选车道线计数和排序 ////////////////////////

void CandSort(std::vector<TrackingLane> &vec)
{
	int k, j, len = vec.size();

	for( k = 1; k < len; k++ )
	{
		TrackingLane temp = vec[k];

		for( j = k - 1; j >= 0 && vec[j].imageAngle > temp.imageAngle; j-- )
			vec[j+1] =  vec[j];

		if( j >= -1 && j != k-1 )
			vec[j+1] = temp;
	}
}

void LaneTrack::UpdateTrackObjs()
{
	const int blen = blocks.size(), clen = trackobjs.size();

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

			cidx = trackobjs.size();
			DpMatch.candi_indexs.push_back(b);
			trackobjs.push_back(line);
		}
	}
}

void LaneTrack::EraseTrackObjs()
{
	for( int i = trackobjs.size()-1; i >= 0; i-- )
	{
		if( trackobjs[i].failed >= trackobjs[i].succeed*5
				|| trackobjs[i].failed >= MAX_LANELINE_FAILED
				|| frameIndex - trackobjs[i].frame > MAX_LANELINE_INTERVAL )
			trackobjs.erase(trackobjs.begin()+i);
	}

	for( int i = trackobjs.size()-1; i > 0; i-- )
	{
		if( fabs(trackobjs[i].imageAngle - trackobjs[i-1].imageAngle) < MIN_CANDILINE_ANGLE )
		{
			if( trackobjs[i].succeed > trackobjs[i-1].succeed )
				trackobjs.erase(trackobjs.begin()+i-1);
			else
				trackobjs.erase(trackobjs.begin()+i);
		}
	}

	while( trackobjs.size() > 10 )
	{
		int max_failed = trackobjs[0].failed, idx = 0;
		for( int i = trackobjs.size()-1; i > 0; i-- )
		{
			if( max_failed < trackobjs[i].failed )
			{
				max_failed = trackobjs[i].failed;
				idx = i;
			}
		}
		trackobjs.erase(trackobjs.begin()+idx);
	}
}


inline void CalcLineByTwoPts(float &k, float &b, const cv::Point2f pt0,
		const cv::Point2f pt1)
{
    float dx = pt0.x - pt1.x;
    float dy = pt0.y - pt1.y;

    k = dx / dy;
    b = pt0.x - pt0.y * k;
}

inline void Polar2KB(const cv::Vec4f src_line, float &k, float &b)
{
	k = src_line[0]/src_line[1];
	b = -k*src_line[3] + src_line[2];
}

inline void CalcLineByTwoPts(cv::Vec4f &dst_line, const cv::Point2f pt0,
		const cv::Point2f pt1)
{
    dst_line[0] = pt0.x - pt1.x;
    dst_line[1] = pt0.y - pt1.y;
    dst_line[2] = (pt0.x + pt1.x)/2;
    dst_line[3] = (pt0.y + pt1.y)/2;

    double delta = sqrt(dst_line[0]*dst_line[0] + dst_line[1]*dst_line[1]);
    if( delta > .001 )
    {
    	delta = 1.0 / delta;
    	dst_line[0] *= delta; dst_line[1] *= delta;
    }
}

inline void CalcPtsByLine(cv::Point2f &pt0, cv::Point2f &pt1,
		const cv::Vec4f src_line, const float step)
{
	float k = src_line[0]/src_line[1], b = -k*src_line[3] + src_line[2];
	float y0 = src_line[3] - step, y1 = src_line[3] + step;
    pt0 = cv::Point2f(k*y0+b, y0);
    pt1 = cv::Point2f(k*y1+b, y1);
}

void TransLine(const cv::Vec4f src_line, cv::Vec4f &dst_line,
		const cv::Mat tran, const float step)
{
	std::vector<cv::Point2f> rts(2), pts(2);
    CalcPtsByLine(pts[0], pts[1], src_line, step);
    cv::perspectiveTransform(pts, rts, tran);
//    std::cout<<pts<<std::endl;
//    std::cout<<rts<<std::endl;
    CalcLineByTwoPts(dst_line, rts[0], rts[1]);
}

void CalcRectifyLine(const StereoRectify *const rectify, const CameraParam &cam,
		cv::Vec4f line, float &k, float &b)
{
    std::vector<cv::Point2f> rts(2), pts(2);
    CalcPtsByLine(pts[0], pts[1], line, 100);
	rectify->toRectifyPoints(pts, rts, &cam);
	CalcLineByTwoPts(k, b, rts[0], rts[1]);
}

void LaneTrack::CalcRectifyLine(const StereoRectify *const rectify,
		float lines[6], const int delay_frames) const
{
    std::vector<cv::Point2f> rts(2), pts(2);
	int left_flag, rght_flag;
	cv::Vec4f left_line, rght_line;

	GetResLines(left_flag, left_line, rght_flag, rght_line, delay_frames);
	lines[2] = left_flag;
	lines[5] = rght_flag;

	if( left_flag != NONE_LINE && fabs(left_line[1]) > .01 )
	{
		::CalcRectifyLine(rectify, rectify->camera->cam_left,
				left_line, lines[0], lines[1]);
//	    CalcPtsByLine(pts[0], pts[1], left_line, 100);
//		rectify->toRectifyPoints(pts, rts, &rectify->camera->cam_left);
//		CalcLineByTwoPts(lines[0], lines[1], rts[0], rts[1]);
	}

	if( rght_flag != NONE_LINE && fabs(rght_line[1]) > .01 )
	{
		::CalcRectifyLine(rectify, rectify->camera->cam_left,
				rght_line, lines[3], lines[4]);
//	    CalcPtsByLine(pts[0], pts[1], rght_line, 100);
//		rectify->toRectifyPoints(pts, rts, &rectify->camera->cam_left);
//		CalcLineByTwoPts(lines[3], lines[4], rts[0], rts[1]);
	}
}

void LaneTrack::FrushStep(const int fast, const int medium, const int slow)
{
	if( frameIndex > procesIndex )
		return;

	if( succeedCount || lineCount )
	{
		double angle = M_PI_2;
		if( leftIndex > -1 )
			angle = MIN(angle, fabs(blocks[leftIndex]->angle - M_PI_2));
		if( rghtIndex > -1 )
			angle = MIN(angle, fabs(blocks[rghtIndex]->angle - M_PI_2));

		if( lineCount == 0 || angle < M_PI_4 - M_PI_10 )
			detectStep = fast;
		else if( lineCount == 2 && angle > M_PI_4 + M_PI_05 )
			detectStep = MIN(MAX_LANELINE_INTERVAL/2, slow);
		else
			detectStep = MIN(MAX_LANELINE_INTERVAL/2, medium);
	}
	else
		detectStep = MIN(MAX_LANELINE_INTERVAL/2, slow);
}


void LaneTrack::RefrushTrackParam()
{
	CalcEndPoint(endpoints[detectCnt % TRACKPARAM_ARR_LENGTH]);

	int epcount = 0;
	endpoint = cv::Point2f(0, 0);

	for( int k = 0; k < TRACKPARAM_ARR_LENGTH; k++ )
	{
		if( endpoints[k] != cv::Point2f(0, 0) )
		{
			endpoint += endpoints[k];
			epcount++;
		}
	}

	if( epcount > 2 )
		endpoint *= 1./(float)epcount;
	else
		endpoint = cv::Point2f(0, 0);

	if( lineCnts[lineCount] < MAX_LINECOUNT )
		lineCnts[lineCount]++;

	for( int i = 2; i >= 0; i-- )
	{
		if( lineCount != i && lineCnts[i] > 0 )
		{
			lineCnts[i]--;
			break;
		}
	}

	if( lineCount == 2 ) // && succeedCount < 50 )
	{
		succeedIndex = frameIndex;
		succeedCount ++;
		failedCount = 0;
	}
	else if( lineCount == 0 )
		failedCount++;

	if( failedCount >= MAX_TRACK_FAILED_COUNT )
		Reset();
}

void LaneTrack::operator()()
{
	const int blen = blocks.size();

	if( detectCnt % 200 == 0 )
		LOGI("lane lines have been detected %d times.", detectCnt);
	procesIndex = frameIndex; detectCnt++;
	lineCount = 0; leftIndex = rghtIndex = -1;

#if DEBUG_PRINT_MESS
	LOGI("===== frame %d, detect %d blocks, %d candidate lines =====",
			frameIndex, blen, (int)trackobjs.size());
//#else
//	LOGI("===== frame %d, detect %d blocks, %d candidate lines =====",
//			frameIndex, blen, (int)trackobjs.size());
#endif

	if( blen )
	{
		flags.resize(blen);
		memset(&flags[0],1, sizeof(flags[0])*blen);

		if( endpoint != cv::Point2f(0, 0) )
			FilterBlocks(endpoint, blocks, &flags[0], 50);

		DpMatch(blocks, trackobjs, &flags[0]);
		RecogLines();
		CheckPassThrough();

#if DEBUG_PRINT_MESS
		for( int i = trackobjs.size()-1; i >= 0; i-- )
		{
			std::stringstream candline_outstr;
			candline_outstr<<trackobjs[i];
			LOGI("candlines[%d]: %s", i, candline_outstr.str().c_str());
		}

		for( int i = blen - 1; i >= 0; i-- )
		{
			std::stringstream block_outstr;
			block_outstr<<*blocks[i];
			LOGI("blocks[%d]: %s", i, block_outstr.str().c_str());
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

		LOGI("left index: %d, rght index: %d", leftIndex, rghtIndex);
#endif

		UpdateTrackObjs();
		CandSort(trackobjs);
	}
	else
	{
		for( int i = trackobjs.size()-1; i >= 0; i-- )
			trackobjs[i].failed++;
	}

	float valid_length = rect.height * 0.8f, max_btm = rect.height;
	_ConfirmLines(valid_length, max_btm);
	_SignResLines(valid_length);
	ConfirmLine(max_btm, valid_length);

	EraseTrackObjs();
	RefrushTrackParam();

//	cv::Vec4f _left_line(0, 0, 0, 0), _rght_Line(0, 0, 0, 0);
	left.flag = rght.flag = NONE_LINE;
	left.CheckLineFlag(left.flag, left.image_line, frameIndex, detectStep*2, min_confirm);
	rght.CheckLineFlag(rght.flag, rght.image_line, frameIndex, detectStep*2, min_confirm);

#if DEBUG_PRINT_MESS
	std::stringstream linecnt_outstr;
	linecnt_outstr<<cv::Mat(1, 3, CV_32SC1, lineCnts);
	LOGI("line count: %s", linecnt_outstr.str().c_str());

	LOGI("detect step: %d, succeed index: %d, count: %d, failed count: %d",
			detectStep, succeedIndex, succeedCount, failedCount);

	std::stringstream lines_str;
	lines_str<<"left: "<<left<<std::endl;
	lines_str<<"rght: "<<rght;
	LOGI("%s", lines_str.str().c_str());
#else
	LOGI("left index: %d, rght index: %d", leftIndex, rghtIndex);
#endif
}

/// 防车道线抖动
void LaneTrack::CheckPassThrough()
{
//	/// 汽车从右边越过车道线，即左边的车道线变成了右边的
//	if( rghtIndex > -1 && blocks[rghtIndex]->angle > M_PI_2 + M_PI_05 )
//	{
//		left.CopyTo(rght);
//		left.Reset();
//	}
//
//	/// 汽车从左边越过车道线，即右边的车道线成成了左边的
//	if( leftIndex > -1 && blocks[leftIndex]->angle < -M_PI_05 )
//	{
//		rght.CopyTo(left);
//		rght.Reset();
//	}
}


void LaneTrack::_SignResLines(const float valid_length)
{
	std::vector<cv::Point2f> ips(2), rps(2);

	if( leftIndex > -1 )
	{
		left.SetLaneLine(*blocks[leftIndex], frameIndex);
		left.mask.at<int>(detectCnt % MAX_LINECOUNT) = 1;

		std::vector<cv::Vec3i> &_vertexs = blocks[leftIndex]->vertexs;
		if( _vertexs.size() == 2 &&
				abs(_vertexs[0][0] - _vertexs[1][0]) > valid_length )
			left.solid = 1;
	}
	else
	{
		left.solid = -1;
		left.prm.angle = 0;
		left.mask.at<int>(detectCnt % MAX_LINECOUNT) = 0;
	}

	if( rghtIndex > -1 )
	{
		rght.SetLaneLine(*blocks[rghtIndex], frameIndex);
		rght.mask.at<int>(detectCnt % MAX_LINECOUNT) = 1;

		std::vector<cv::Vec3i> &_vertexs = blocks[rghtIndex]->vertexs;
		if( _vertexs.size() == 2 &&
				abs(_vertexs[0][0] - _vertexs[1][0]) > valid_length )
			rght.solid = 1;
	}
	else
	{
		rght.solid = -1;
		left.prm.angle = 0;
		rght.mask.at<int>(detectCnt % MAX_LINECOUNT) = 0;
	}
}

void LaneTrack::_ConfirmLines(const float valid_length, const float max_btm)
{
	const int confirm_idx = detectCnt % MAX_LINECOUNT;

	if( leftIndex > -1 )
	{
		left.top.at<int>(confirm_idx) = blocks[leftIndex]->head->row;
		left.btm.at<int>(confirm_idx) = blocks[leftIndex]->tear->row;
	}
	else
	{
		left.top.at<int>(confirm_idx) = max_btm;
		left.btm.at<int>(confirm_idx) = 0;
	}

	if( rghtIndex > -1 )
	{
		const LaneBlock *block = blocks[rghtIndex];
		rght.top.at<int>(confirm_idx) = block->head->row;
		rght.btm.at<int>(confirm_idx) = block->tear->row;
	}
	else
	{
		rght.top.at<int>(confirm_idx) = max_btm;
		rght.btm.at<int>(confirm_idx) = 0;
	}
}

void LaneTrack::ConfirmLine(const float max_btm, const float valid_length)
{
	double top = max_btm, btm = 0;
	cv::minMaxIdx(left.top, &top, 0);
	cv::minMaxIdx(left.btm, 0, &btm);

	if( btm - top < valid_length )
		left.confirmed = 0;
	else if( leftIndex > -1 )
		left.confirmed++;

	top = max_btm, btm = 0;
	cv::minMaxIdx(rght.top, &top, 0);
	cv::minMaxIdx(rght.btm, 0, &btm);

	if( btm - top < valid_length )
		rght.confirmed = 0;
	else if( rghtIndex > -1 )
		rght.confirmed++;
}

void LaneTrack::GetResLines(int &_left_flag, cv::Vec4f &_left_line,
		int &_rght_flag, cv::Vec4f &_rght_line, int delay_frames) const
{
	if( delay_frames <= 0 )
		delay_frames = detectStep;
	_left_flag = _rght_flag = NONE_LINE;

	left.CheckLineFlag(_left_flag, _left_line, frameIndex, delay_frames, min_confirm);
	rght.CheckLineFlag(_rght_flag, _rght_line, frameIndex, delay_frames, min_confirm);
}


static void DrawLaneLine(cv::Mat image, const int flag, const cv::Vec4f line,
		const int width = 2)
{
	const cv::Scalar red(0, 0, 255), blue(255, 0, 0), green(0, 255, 0);

	if( flag&LaneTrack::DETECTED_LINE )
	{
		const cv::Scalar color = flag&LaneTrack::RUNIN_LINE ? red : blue;
		DrawSolidLine(image, line, color, width);
		if( flag&LaneTrack::CONFIRM_LINE )
			DrawDottedCircle(image, line, blue, 5, 2, 0);
	}
	else if( flag&LaneTrack::CALCULATED_LINE )
	{
		const cv::Scalar color = flag&LaneTrack::RUNIN_LINE ? red : green;
		DrawSolidLine(image, line, color, width);
		if( flag&LaneTrack::CONFIRM_LINE )
			DrawDottedCircle(image, line, green, 5, 2, 0);
	}
}

static void DrawFixedCircle(cv::Mat image, const cv::Vec4f line, const int y,
		const cv::Scalar color, const int width = -1)
{
	const float k = line[0]/line[1], b = -k*(line[3]) + line[2];
	cv::circle(image, cv::Point(k*y+b, y), 10, color, width);
}

void LaneTrack::Show(cv::Mat image)
{
	const cv::Scalar red(0, 0, 255), yellow(0, 255, 255);

	int _left_flag = NONE_LINE, _rght_flag = NONE_LINE, linewidth = 2;
	cv::Vec4f _left_line(0, 0, 0, 0), _rght_Line(0, 0, 0, 0);

	GetResLines(_left_flag, _left_line, _rght_flag, _rght_Line, detectStep*2);
	DrawLaneLine(image, _left_flag, _left_line, linewidth);
	DrawLaneLine(image, _rght_flag, _rght_Line, linewidth);

	if( _left_flag != NONE_LINE )
		DrawFixedCircle(image, _left_line, image.rows - 100, red);
	if( _rght_flag != NONE_LINE )
		DrawFixedCircle(image, _rght_Line, image.rows - 100, yellow);

	cv::circle(image, endpoint,  3, red, -1);
	cv::circle(image, endpoint, 10, red,  2);
}



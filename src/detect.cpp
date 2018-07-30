#include <vector>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define LOG_TAG "LaneDetect detect.cpp"
#include "core/utils.hpp"
#include "track.hpp"


void ShiftDiff(const cv::Mat1s src, const int sx, const int sy, cv::Mat1s &dst)
{
	if( dst.size() != src.size() )
		dst.create(src.size());
	dst.setTo(0);

	for( int i = sy; i < src.rows - sy; i++ )
	{
		const short* sd0 = src.ptr<short>(i-sy);
		const short* sd1 = src.ptr<short>(i+sy);
		short* dd = dst.ptr<short>(i);

		for( int j = sx; j < src.cols-sx; j++ )
			dd[j] = sd1[j+sx] - sd0[j-sx];
	}
}

void CustomSum(const cv::Mat1b src, const int bw, const int bh, cv::Mat1s &dst)
{
	cv::Mat sumkr = cv::getStructuringElement(0, cv::Size(bw, bh));
	cv::filter2D(src, dst, CV_16S, sumkr);
}

cv::Mat1s GenEdgeImage(const cv::Mat src, const cv::Size size)
{
	cv::Mat1s edge, _sum(src.size());
	cv::Mat gray = src;

	if( size.width < 1 || size.width > 32 || size.height < 1 || size.height > 32 )
	{
		LOGI("##Error: block size for detect edge is wrong.");
		return edge;
	}
	if( src.channels() != 3 && src.channels() != 1 )
	{
		LOGI("##Error: source image channel must be 3 or 1.");
		return edge;
	}

	if( src.channels() == 3 )
		cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	CustomSum(src, size.width+1, size.height, _sum);
	ShiftDiff(_sum, 2, 0, edge);

	return edge;
}

void GenEdgeImage(const cv::Mat src, const cv::Size size, cv::Mat1s &edge)
{
	if( size.width < 1 || size.width > 32 || size.height < 1 || size.height > 32 )
	{
		LOGI("##Error: block size for detect edge is wrong.");
		return;
	}
	if( src.channels() != 3 && src.channels() != 1 )
	{
		LOGI("##Error: source image channel must be 3 or 1.");
		return;
	}

	cv::Mat1s _sum(src.size());
	cv::Mat gray = src;

	if( src.channels() == 3 )
		cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	CustomSum(src, size.width+1, size.height, _sum);
	ShiftDiff(_sum, 2, 0, edge);
}

static void GenRectifyMap(cv::Mat &mapx, cv::Mat &mapy, const StereoRectify *rectify)
{
	int dw = rectify->size.width, dh = rectify->size.height;

	mapx = cv::Mat(dh, dw*2, CV_32FC1);
	mapy = cv::Mat(dh, dw*2, CV_32FC1);

	rectify->maps[0].copyTo(mapx(rectify->left));
	rectify->maps[1].copyTo(mapy(rectify->left));

	cv::Mat temp = rectify->maps[2] + rectify->camera->imageSize.width;
	temp.copyTo(mapx(rectify->rght));
	rectify->maps[3].copyTo(mapy(rectify->rght));
}

StereoPre::StereoPre(const Birdview &bird, const size_t bw, const size_t bh):
		top(bird.end_point.y - 32), blockw(bw), blockh(bh)
{
	const StereoRectify *rectify = bird.rectify;
	const int rech = MIN(MAX_SCAN_IMAGE_ROWS, rectify->size.height - top), recw = rectify->size.width;
	const int camh = rectify->camera->imageSize.height, camw = rectify->camera->imageSize.width;

	const cv::Rect rect_rec(0, top, recw*2, rech);
	cv::Mat _mapx, _mapy;

	src.create(camh, camw*2);
	rec.create(rech, recw*2);
	edge.create(rech, recw);
	_sum.create(rech, recw);

	GenRectifyMap(_mapx, _mapy, rectify);
	rmapx = _mapx(rect_rec).clone();
	rmapy = _mapy(rect_rec).clone();

	left = cv::Rect(0, 0, recw, rech);
	rght = cv::Rect(recw, 0, recw, rech);

	edge.setTo(0);
}

int StereoPre::operator()(const cv::Mat srcimg)
{
	const size_t w = blockw, h = blockh;
	cv::Mat gray, _rec = rec(left);

	cv::cvtColor(srcimg, gray, cv::COLOR_BGR2GRAY);
	cv::remap(gray, rec, rmapx, rmapy, cv::INTER_LINEAR);

	CustomSum(_rec, w+1, 2*h+1, _sum);
	ShiftDiff(_sum, 2, 0, edge);

	return 0;
}


static float fit_line(cv::Vec4f &line, const float x, const float y,
		const float x2, const float y2, const float xy)
{
	float dx2 = x2 - x * x;
	float dy2 = y2 - y * y;
	float dxy = xy - x * y;

    float t = (float) atan2( 2 * dxy, dx2 - dy2 ) / 2;
    line[0] = (float) cos( t );
    line[1] = (float) sin( t );

    line[2] = (float) x;
    line[3] = (float) y;

    return t < 0 ? t + CV_PI : t;
}

class LaneBlockParam : public LaneBlock
{
	friend class BlockDetectBody;

	double my1, my2, mxlr, mx1[3], mx2[3], mxy[3]; // line moment

public:

	LaneBlockParam(): LaneBlock(0), my1(0), my2(0), mxlr(0)
	{
		Init();
	}

	LaneBlockParam(const int idx, EdgeLine *cel): LaneBlock(idx),
			my1(0), my2(0), mxlr(0)
	{
		Init();
		lineCnt = 1;
		pointCnt = cel->rght - cel->left;
		head = tear = cel;
		cel->index = idx;
		Moment(cel->row, cel->left, cel->rght);
	}

	void Init()
	{
		vertexs.clear();
		my1 = 0; my2 = 0;
		mx1[0] = mx1[1] = mx1[2] = 0;
		mx2[0] = mx2[1] = mx2[2] = 0;
		mxy[0] = mxy[1] = mxy[2] = 0;
	}

	void Append(EdgeLine *cel)
	{
		const int w = cel->rght - cel->left;

		pointCnt += w;
		lineCnt++;
		cel->index = index;

		if( head && head->row > cel->row )
		{
			cel->next = head;
			head = cel;
		}
		else if( head && tear )
		{
			tear->next = cel;
			tear = cel;
		}
		else
			head = tear = cel;

		Moment(cel->row, cel->left, cel->rght);
	}

	void Remove(EdgeLine *cel)
	{
		pointCnt -= cel->rght - cel->left;
		lineCnt --;

		my1 -= cel->row;
		mx1[0] -= cel->left; mx1[1] -= cel->rght;
		my2 -= cel->row*cel->row;
		mx2[0] -= cel->left*cel->left; mx2[1] -= cel->rght*cel->rght;
		mxlr -= cel->left*cel->rght;
		mxy[0] -= cel->left*cel->row; mxy[1] -= cel->rght*cel->row;
	}

	void Moment(const int y, const int lx, const int rx)
	{
		my1 += y;
		mx1[0] += lx; mx1[1] += rx;
		my2 += y*y;
		mx2[0] += lx*lx; mx2[1] += rx*rx;
		mxlr += lx*rx;
		mxy[0] += lx*y; mxy[1] += rx*y;
	}

private:

	void CalcParam()
	{
		if( lineCnt < 2 )
			return;
		const float s = 1./lineCnt;

		mx1[2] = (mx1[0] + mx1[1])/ 2;
		mx2[2] = (mx2[0] + mx2[1] + 2*mxlr)/4;
		mxy[2] = (mxy[0] + mxy[1])/ 2;

		fit_line(left_line, mx1[0]*s, my1*s, mx2[0]*s, my2*s, mxy[0]*s);
		fit_line(rght_line, mx1[1]*s, my1*s, mx2[1]*s, my2*s, mxy[1]*s);
		angle = fit_line(center_line, mx1[2]*s, my1*s, mx2[2]*s, my2*s, mxy[2]*s);

		vertexs.push_back(cv::Vec3i(head->left, head->rght, head->row));
		vertexs.push_back(cv::Vec3i(tear->left, tear->rght, tear->row));
	}

	void Merge(LaneBlockParam &block)
	{
		lineCnt += block.lineCnt;
		pointCnt += block.pointCnt;

		EdgeLine *el = block.head;
		while( el )
		{
			el->index = index;
			el = el->next;
		}

		if( head->row < block.head->row )
		{
			tear->next = block.head;
			tear = block.tear;
		}
		else
		{
			block.tear->next = head;
			head = block.head;
			std::swap(vertexs, block.vertexs);
		}

		if( block.vertexs[0][2] - vertexs[vertexs.size()-1][2] < 5 )
		{
			vertexs.erase(vertexs.end());
			block.vertexs.erase(block.vertexs.begin());
		}
		vertexs.insert(vertexs.end(), block.vertexs.begin(), block.vertexs.end());

		my1 += block.my1;
		my2 += block.my2;

		mx1[0] += block.mx1[0];
		mx1[1] += block.mx1[1];
		mx1[2] += block.mx1[2];

		mx2[0] += block.mx2[0];
		mx2[1] += block.mx2[1];
		mx2[2] += block.mx2[2];

		mxy[0] += block.mxy[0];
		mxy[1] += block.mxy[1];
		mxy[2] += block.mxy[2];

		const float s = 1./ lineCnt;
		fit_line(left_line, mx1[0]*s, my1*s, mx2[0]*s, my2*s, mxy[0]*s);
		fit_line(rght_line, mx1[1]*s, my1*s, mx2[1]*s, my2*s, mxy[1]*s);
		angle = fit_line(center_line, mx1[2]*s, my1*s, mx2[2]*s, my2*s, mxy[2]*s);

		block.pointCnt = block.lineCnt = 0;
		block.vertexs.clear();
	}

	bool CheckInLine(const LaneBlockParam block, const float minds = 5)
	{
		bool res = false;

		// 平均宽度小的，不能接在下面
		if( tear->row < block.head->row )
			res = pointCnt*block.lineCnt <= block.pointCnt*lineCnt;
		else if( head->row > block.tear->row )
			res = pointCnt*block.lineCnt >= block.pointCnt*lineCnt;

		return  res && //line_distance(center_line, block.center_line) < 2 &&
				line_distance(left_line, block.left_line) < minds &&
				line_distance(rght_line, block.rght_line) < minds;
	}

	/// 对矩形区域的判定条件
	bool CheckRectBlock(const int stripe)
	{
		const int y0 = head->row, y1 = tear->row, rows = y1 - y0 + 1;
		bool res = 1;

		res &= rows > MIN_LANEBLOCK_LINECOUNT;             ///< 至少有24行
		res &= lineCnt*stripe  >  MIN_LANEBLOCK_LINECOUNT; ///< 至少检测到24条线
		res &= pointCnt*stripe > MIN_LANEBLOCK_PIXELCOUNT; ///< 像素个数超过240

		if( res )
		{
			float k = left_line[0]/left_line[1], b = left_line[2];
			float lx0 = k*(y0 - left_line[3]) + b, lx1 = k*(y1 - left_line[3]) + b;
			k = rght_line[0]/rght_line[1], b = rght_line[2];
			float rx0 = k*(y0 - rght_line[3]) + b, rx1 = k*(y1 - rght_line[3]) + b;

			slope = (rx1 - lx1 - rx0 + lx0) / (y1 - y0);
			res &= slope > MIN_LANEBLOCK_SLOPE;
			res &= slope < MAX_LANEBLOCK_SLOPE;  ///< 车道线区域宽度变大的斜率在（0.10， 0.45）区间内
		}

		return res;
	}
};

class BlockDetectBody: public LaneBlockDetect
{
	const float thres, min_line_dst;
	std::vector<int> flags;

	std::vector<std::vector<EdgeLine> > line_tables;
	std::vector<LaneBlock> track_blocks;
	std::vector<LaneBlockParam> all_blocks;

public:

	BlockDetectBody(const float thres, const float min_lds): LaneBlockDetect(),
		thres(thres), min_line_dst(min_lds), track_blocks(12), all_blocks(64)
	{}

	BlockDetectBody(const DetectParam *stripe, const float thres, const float min_lds):
		LaneBlockDetect(), thres(thres), min_line_dst(min_lds), track_blocks(12), all_blocks(64)
	{
		rowStep = *stripe;
	}

	void operator()(const std::vector<cv::Vec4f> lines, const cv::Mat1s edgeImg);

	void operator()(const cv::Mat1s edgeImg);

	int SetDetectRegion(const cv::Size imageSize, const cv::Mat road2image,
			const float width, const float height, const float dist1 = 5, const float dist2 = 50);

	void SetDetectRegion(const cv::Size size, const int bottom, const int top);

	void SetStripe(DetectParam *_stripe)
	{
//		if( stripe->stripe > _stripe->stripe )
//		{
//			for( int row = 0; row < rect.height; row++ )
//				line_tables[row].clear();
//		}
		if( rowStep.stripe != _stripe->stripe )
			rowStep = *_stripe;
	}

	void Clear()
	{
		for( int row = 0; row < rect.height; row++ )
			line_tables[row].clear();
		track_blocks.clear();
		lane_blocks.clear();
		all_blocks .clear();
	}

	void Show(cv::Mat dspImage, const uchar *flags = 0) const;

	void Show(cv::Mat dspImage, const int block_idx, const int color = 10) const;

private:

	void ScanEdgeLines(const cv::Mat1s edge);

	void RetrieveBlocks(LaneBlock &block, int &blockCnt, const cv::Vec4f line);

	void RetrieveBlocks(std::vector<LaneBlockParam> &all_blocks);

	void MergeBlocks();
};

cv::Ptr<LaneBlockDetect> LaneBlockDetect::CreateDetect(const float thres, const float min_lds)
{
	return cv::makePtr<BlockDetectBody>(thres, min_lds);
}

cv::Ptr<LaneBlockDetect> LaneBlockDetect::CreateDetect(const DetectParam *stripe,
		const float thres, const float min_lds)
{
	return cv::makePtr<BlockDetectBody>(stripe, thres, min_lds);
}

int BlockDetectBody::SetDetectRegion(const cv::Size imageSize, const cv::Mat road2image,
		const float width, const float height, const float dist1, const float dist2)
{
	const float max_height = height <= 0 ? imageSize.height - 20 :
			MIN(imageSize.height - 20, height);
	const float min_top = imageSize.height/2 - 100;
	float top = imageSize.height, btm = 0;
	int res = STEREO_SUCCEED;

	if( !road2image.empty() )
	{
		std::vector<cv::Point2f> rps(4), ips(4);

		rps[0] = cv::Point2f(-width, dist1);
		rps[1] = cv::Point2f(+width, dist1);
		rps[2] = cv::Point2f(+width*2.0, dist2);
		rps[3] = cv::Point2f(-width*2.0, dist2);
		cv::perspectiveTransform(rps, ips, road2image);

		top = MIN(ips[3].y, ips[2].y);
		btm = MAX(ips[0].y, ips[1].y);
//		LOGI("top: %f, btm: %f, max_height: %f.", top, btm, max_height);

		if( btm > max_height )
			btm = max_height;
		if( top < min_top )
			top = min_top;
	}

	if( top > max_height || btm < imageSize.height/2 - 40 || top >= btm )
	{
		res = road2image.empty() ? STEREO_ERR_NOSRC : STEREO_ERR_PARAM;
		btm = max_height;
		top = imageSize.height/2 - 40;
	}

	rect = cv::Rect(0, top, imageSize.width, btm - top);
//	rect = cv::Rect(0, top/2, imageSize.width/2, (btm - top)/2);
	line_tables.resize(rect.height);

	return res;
}


void BlockDetectBody::SetDetectRegion(const cv::Size imageSize,
		const int btm, const int top)
{
	rect = cv::Rect(0, top, imageSize.width, btm - top);
	line_tables.resize(rect.height);
}

void BlockDetectBody::RetrieveBlocks(LaneBlock &block, int &blockCnt, const cv::Vec4f line)
{
	const float k = line[0] / line[1], neighbor = 2;
	const float b = -k*line[3] + line[2];

	std::vector<LaneBlockParam> vtblocks;
	LaneBlockParam curr_block;
	int pre_row = -1;
	EdgeLine *pel = 0;

	for( int y = 0; y < rect.height; y++ )
	{
		const float x = k*y + b;
		std::vector<EdgeLine> &clines = line_tables[y];

		for( int i = clines.size()-1; i >= 0; i-- )
		{
			EdgeLine &cel = clines[i];
			if( cel.index > -1 || cel.left > x + neighbor || cel.rght < x - neighbor )
				continue;

			if( pel != 0 && !rowStep.CheckSame(*pel, cel, pre_row, y) )
			{
				if( curr_block.lineCnt > 2 )
					vtblocks.push_back(curr_block);
				curr_block = LaneBlockParam(++blockCnt, &cel);
			}
			else if( pel == 0 )
				curr_block = LaneBlockParam(++blockCnt, &cel);
			else
				curr_block.Append(&cel);

			cel.index = curr_block.index;
			pel = &cel;
			pre_row = y;
			break;
		}
	}

	if( curr_block.lineCnt > 2 )
		vtblocks.push_back(curr_block);

	const int len = vtblocks.size();
	if( len == 0 )
		return;
	int max_count = 10;
	LaneBlockParam *cblock = &vtblocks[0];

	vtblocks[0].CalcParam();
	for( int i = 1; i < len; i++ )
	{
		vtblocks[i].CalcParam();
		if( cblock->CheckInLine(vtblocks[i], min_line_dst*2) )
			cblock->Merge(vtblocks[i]);
		else
			cblock = &vtblocks[i];
	}

	for( int i = 0; i < len; i++ )
	{
		if( max_count < vtblocks[i].pointCnt && vtblocks[i].CheckRectBlock(rowStep.stripe) )
		{
			max_count = vtblocks[i].pointCnt;
			block = vtblocks[i];
		}
	}
}

static float EdgeLineDist(const EdgeLine *el1, const EdgeLine *el2)
{
	const float dx = (el1->left + el1->rght - el2->left - el2->rght)*.5;
	const float dw = el1->rght - el1->left - el2->rght + el2->left;
	const float dy = el1->row - el2->row;
	return dx*dx + dy*dy + dw*dw;
}

// 对候选区域进行分块
void BlockDetectBody::RetrieveBlocks(std::vector< LaneBlockParam > &all_blocks)
{
	float min_dist = 1e32;
	int find_index = 0, pre_row = 0;
	EdgeLine *line = 0;
	LaneBlockParam *block = 0;

	for( int row = 1; row < rect.height; row++ )
	{
		const int _end = MAX(0, row - rowStep.max_interv);
		std::vector<EdgeLine> &curr_lines = line_tables[row];
		const int clen = curr_lines.size();

		for( int j = 0; j < clen; j++ )
	    {
			EdgeLine &curr = curr_lines[j];
			if( curr.index )
				continue;
			line = 0; min_dist = 1e32;
			block = 0;
			find_index = 0, pre_row = row - 1;

			for(; pre_row >= _end && line == 0; pre_row-- )
			{
				std::vector<EdgeLine> &prev_lines = line_tables[pre_row];

				for( int m = prev_lines.size()-1; m >= 0; m-- )
				{
					if( prev_lines[m].index < 0x10000
							&& rowStep.CheckSame(prev_lines[m], curr) )
					{
						float dist = EdgeLineDist(&curr, &prev_lines[m]);
						if( min_dist > dist )
						{
							min_dist = dist;
							line = &prev_lines[m];
						}
					}
				}
			}

			if( line && line->index )
			{
				find_index = line->index;
				block = &all_blocks[find_index-1];
				EdgeLine *next = line->next;

				if( next )
				{
					float dist = EdgeLineDist(line, next);
					if( dist > min_dist )
					{
						int index = all_blocks.size()+ 1;
						LaneBlockParam l(index, next);
						all_blocks.push_back(l);
						block->Remove(next);

						while( next->next )
						{
							next = next->next;
							all_blocks[index-1].Append(next);
							block->Remove(next);
						}
						block->tear = line;
					}
					else
					{
						find_index = 0;
						line = 0;
					}
				}
				else
					block->Append(&curr);
			}

			if( find_index == 0 )
			{
				find_index = all_blocks.size() + 1;
				LaneBlockParam l(find_index, &curr);
				all_blocks.push_back(l);
			}
	    }
	}

	for( int j = all_blocks.size() - 1; j >= 0; j-- )
		all_blocks[j].CalcParam();
}

void BlockDetectBody::MergeBlocks()
{
	const int alen = all_blocks.size();
	flags.resize(alen);
	memset(&flags[0], 1, sizeof(int)*alen);

	for( int j = alen-1; j >= 0; j-- )
	{
		LaneBlockParam &block = all_blocks[j];
		flags[j] = block.lineCnt > 1;
		if( !flags[j] )
			continue;

		for( int k = j+1; k < alen; k++ )
		{
			if( flags[k] && block.CheckInLine(all_blocks[k], min_line_dst) )
			{
				block.Merge(all_blocks[k]);
				flags[k] = 0;
			}
		}
	}

	// 过滤掉非矩形区域
	for( int j = 0; j < alen; j++ )
	{
		LaneBlockParam *block = &all_blocks[j];
		int find = 0;

//#if DEBUG_PRINT_MESS
//		if( flags[j] && block->totalCnt > 20 && block->count > 200 )
//			std::cout<<"block["<<j<<"]: "<<*block<<std::endl;
//#endif
		if( !flags[j] || !block->CheckRectBlock(rowStep.stripe) )
			continue;

		for( int k = lane_blocks.size()-1; k >= 0 && !find; k-- )
		{
			const LaneBlock *prev = lane_blocks[k];

			if( fabs(prev->angle - block->angle) < M_PI_05/2
				&& line_distance(prev->center_line, block->center_line) < MIN_LANEBLOCK_DISTANCE )
			{
				if( prev->pointCnt < block->pointCnt )
					lane_blocks[k] = block;
				find = 1;
			}
		}

		if( find == 0 )
			lane_blocks.push_back((LaneBlock*)block);
	}
}

inline void ShiftLaneBlocks(std::vector< LaneBlock* > &blocks, const float y)
{
	for( int i = blocks.size()-1; i >= 0; i-- )
	{
		blocks[i]->center_line[3] += y;
//		blocks[i]->center_line[2] *= 2;
//		blocks[i]->center_line[3] *= 2;
	}
}

inline void LaneSort(std::vector<LaneBlock*> &vec)
{
	int k, j, len = vec.size();

	for( k = 1; k < len; k++ )
	{
		LaneBlock *temp = vec[k];

		for( j = k-1; j >= 0 && ((LaneBlock*)vec[j])->angle > temp->angle; j-- )
			vec[j+1] =  vec[j];

		if( j >= -1 && j != k-1 )
			vec[j+1] = temp;
	}
}

#define MinEdge(ed, j, start) (\
		ed[j] < ed[j-2] && ed[j] < ed[j-1] && ed[j-1] < ed[j-2] && \
		ed[j] < ed[j+1] && ed[j] < ed[j+2] && ed[j+1] < ed[j+2] )

#define MaxEdge(ed, j, start) (\
		ed[j] > ed[j-2] && ed[j] > ed[j-1] && ed[j-1] > ed[j-2] && \
		ed[j] > ed[j+1] && ed[j] > ed[j+2] && ed[j+1] > ed[j+2] )

void BlockDetectBody::ScanEdgeLines(const cv::Mat1s edge)
{
	const float thresh = thres > 0 ? thres : EDGE_GRADIENT_THRESH * 12;

	for( int row = 0; row < edge.rows; row += rowStep.stripe )
	{
		const short *ed = edge.ptr<short>(row);
		int start = 0, _left = 2, _rght = edge.cols - 2;

		for( int j = _left; j < _rght; j++ )
		{
			if( start && ed[j] < -thresh && MinEdge(ed, j, start) )
			{
				if( j - start < MAX_LANEBLOCK_WIDTH
						&& j - start > MIN_LANEBLOCK_WIDTH )
				{
					EdgeLine el(start, j, row);
					line_tables[row].push_back(el);
				}
				start = 0;
			}
			else if( start && j-start >= MAX_LANEBLOCK_WIDTH )
				start = 0;
			else if( !start && ed[j] > thresh && MaxEdge(ed, j, start) )
				start = j;
		}
	}
}


#if !ANDROID && DEBUG_ALGORITHM
static void ColorLane(const std::vector< std::vector<EdgeLine> > &line_tables,
		cv::Mat image, const int block_idx, const int color_idx);
static void ColorLineTable(cv::Mat image, const cv::Size size,
		const std::vector< std::vector<EdgeLine> > &line_tables);
static void ColorLineTable(cv::Mat image, const cv::Size size,
		const std::vector< std::vector<EdgeLine> > &line_tables,
		const std::vector<LaneBlockParam > &all_blocks, const int min_lineCnt);
static void DrawCenterLine(cv::Mat image, const std::vector<LaneBlock*> &blocks,
		const int width);
#endif


void BlockDetectBody::operator()(const cv::Mat1s edge)
{
	Clear();

	ScanEdgeLines(edge);
	RetrieveBlocks(all_blocks);

#if !ANDROID && DEBUG_ALGORITHM
	cv::Mat dimg, mimg, abs_edge = cv::abs(edge);
	abs_edge.convertTo(dimg, CV_8U);
	abs_edge.convertTo(mimg, CV_8U);

	if( edge.channels() == 1 )
	{
		cv::cvtColor(dimg, dimg, cv::COLOR_GRAY2BGR);
		cv::cvtColor(mimg, mimg, cv::COLOR_GRAY2BGR);
	}

	ColorLineTable(dimg, rect.size(), line_tables);
#endif

	MergeBlocks();
	LaneSort(lane_blocks);

#if !ANDROID && DEBUG_ALGORITHM
	ColorLineTable(mimg, rect.size(), line_tables, all_blocks, 10);
	DrawCenterLine(mimg, lane_blocks, 3);
	cv::imshow("blocks", ConcatImages(dimg, mimg));
#endif

	ShiftLaneBlocks(lane_blocks, rect.y);
}

void BlockDetectBody::operator()(const std::vector<cv::Vec4f> prev_lines, const cv::Mat1s edge)
{
	int blockCnt = 0x10000;

	track_blocks.clear();
	Clear();
	ScanEdgeLines(edge);

#if !ANDROID && DEBUG_ALGORITHM
	const cv::Scalar red(0, 0, 255), blue(255, 0, 0), yellow(0, 255, 255);
	cv::Mat eimg = edge.clone(), bimg = edge.clone();
//	ColorLineTable(eimg, rect.size(), line_tables);

	for( int i = 0; i < rect.height; i++ )
	{
		std::vector<EdgeLine> &lines = line_tables[i];

		for( int j = lines.size()-1; j >= 0; j-- )
		{
			const cv::Point p1(lines[j].left, i), p2(lines[j].rght, i);
			cv::line(eimg, p1, p2, blue);
		}
	}
#endif

	for( int i = prev_lines.size()-1; i >= 0; i-- )
	{
		LaneBlock block = LaneBlock();
		cv::Vec4f _line = prev_lines[i];
		_line[3] -= rect.y;

		RetrieveBlocks(block, blockCnt, _line);
		if( block.pointCnt > 40 )
			track_blocks.push_back(block);

#if !ANDROID && DEBUG_ALGORITHM
		ColorLane(line_tables, bimg, block.index, 9);
		DrawSolidLine(bimg, _line, yellow, 2);
#endif
	}

	RetrieveBlocks(all_blocks);
//	LOGI("detect all blocks: %d", (int)all_blocks.size());

#if !ANDROID && DEBUG_ALGORITHM
	cv::Mat dimg = edge.clone(), mimg = edge.clone();
	ColorLineTable(dimg, rect.size(), line_tables);
#endif

	MergeBlocks();
	int tlen = track_blocks.size();
	for( int i = 0; i < tlen; i++ )
		lane_blocks.push_back(&track_blocks[i]);
	LaneSort(lane_blocks);

#if !ANDROID && DEBUG_ALGORITHM
	ColorLineTable(mimg, rect.size(), line_tables);
	DrawCenterLine(mimg, lane_blocks, 1);

	cv::Mat topimg = ConcatImages(eimg, bimg);
	cv::Mat btmimg = ConcatImages(dimg, mimg);
	cv::imshow("blocks", ConcatImages (topimg, btmimg, 1) );
#endif

	ShiftLaneBlocks(lane_blocks, rect.y);
}


const int color_number = 12;
const static cv::Vec3b colors[] = {
		cv::Vec3b(0, 0, 255), cv::Vec3b(0, 255, 0), cv::Vec3b(255, 0, 0),
		cv::Vec3b(0, 255, 255), cv::Vec3b(255, 0, 255), cv::Vec3b(255, 255, 0),
		cv::Vec3b(128, 255, 0), cv::Vec3b(255, 128, 0), cv::Vec3b(0, 255, 128),
		cv::Vec3b(0, 128, 255), cv::Vec3b(255, 0, 128), cv::Vec3b(128, 0, 255)
};

void LaneBlock::Show(cv::Mat image, const int color_idx) const
{
	const cv::Scalar color = colors[color_idx%color_number];
	EdgeLine *el = head;

	while( el )
	{
		cv::Point p1(el->left, el->row), p2(el->rght, el->row);
		cv::line (image, p1, p2, color);
		el = el->next;
	}
}

void BlockDetectBody::Show(cv::Mat dspImage, const int block_idx, const int color) const
{
	cv::Mat image = dspImage(rect);
#if WITH_LINE_LABEL
	ColorLane(line_tables, image, lane_blocks[block_idx].index, color);
#else
	if( block_idx > -1 )
		lane_blocks[block_idx]->Show(image, color);
#endif
}

void BlockDetectBody::Show(cv::Mat dspImage, const uchar *flags) const
{
	int color_idx = 3, blen = lane_blocks.size();
	cv::Mat image = dspImage(rect);

	if( flags != 0 )
	{
		for( int i = 0; i < blen; i++ )
		{
			if( flags[i] && ((LaneBlock*)lane_blocks[i])->pointCnt > 20 )
			{
#if WITH_LINE_LABEL
				ColorLane(line_tables, image, lane_blocks[i].index, color_idx);
#else
				lane_blocks[i]->Show(image, color_idx++);
#endif
			}
		}
	}
	else
	{
		for( int i = 0; i < blen; i++ )
		{
			if( ((LaneBlock*)lane_blocks[i])->pointCnt > 20 )
			{
#if WITH_LINE_LABEL
				ColorLane(line_tables, image, lane_blocks[i].index, color_idx);
#else
				lane_blocks[i]->Show(image, color_idx++);
#endif
			}
		}
	}

	cv::rectangle(dspImage, rect, cv::Scalar(0, 255, 255));
}


#if WITH_LINE_LABEL || (!ANDROID && DEBUG_ALGORITHM)
static void ColorLane(const std::vector< std::vector<EdgeLine> > &line_tables,
		cv::Mat image, const int block_idx, const int color_idx)
{
	cv::Vec3b color = colors[color_idx%color_number];

	for( int row = 0; row < image.rows; row++ )
	{
		const std::vector<EdgeLine> &lines = line_tables[row];

		for( int col = lines.size()-1; col >= 0; col-- )
		{
			if( lines[col].index == block_idx )
			{
				const cv::Point p1(lines[col].left, row), p2(lines[col].rght, row);
				cv::line(image, p1, p2, color);
			}
		}
	}
}
#endif

#if !ANDROID && DEBUG_ALGORITHM
static void ColorLineTable(cv::Mat image, const cv::Size size,
		const std::vector< std::vector<EdgeLine> > &line_tables)
{
	if( image.empty() || image.size() != size )
		image = cv::Mat::zeros(size, CV_8UC3);

	for( int row = 0; row < image.rows; row++ )
	{
		const std::vector<EdgeLine> &cls = line_tables[row];

		for( int k = cls.size()-1; k >= 0; k-- )
		{
			cv::Point p1(cls[k].left, row), p2(cls[k].rght, row);
			cv::line(image, p1, p2, colors[cls[k].index%color_number]);
		}
	}
}

static void ColorLineTable(cv::Mat image, const cv::Size size,
		const std::vector< std::vector<EdgeLine> > &line_tables,
		const std::vector<LaneBlockParam> &all_blocks, const int min_lineCnt)
{
	if( image.empty() || image.size() != size )
		image = cv::Mat::zeros(size, CV_8UC3);

	for( int row = 0; row < image.rows; row++ )
	{
		const std::vector<EdgeLine> &cls = line_tables[row];

		for( int k = cls.size()-1; k >= 0; k-- )
		{
			if( all_blocks[cls[k].index-1].lineCnt < min_lineCnt )
				continue;
			cv::Point p1(cls[k].left, row), p2(cls[k].rght, row);
			cv::line(image, p1, p2, colors[cls[k].index%color_number]);
		}
	}
}

static void DrawCenterLine(cv::Mat image, const std::vector<LaneBlock*> &blocks,
		const int width)
{
	for( int i = blocks.size() - 1; i >= 0; i-- )
		DrawSolidLine(image, blocks[i]->center_line,
				cv::Scalar(0, 255, 255), width);
}
#endif


//int BlockDetectBody::SetDetectRegion(const cv::Size imageSize, const cv::Mat road2image,
//		const float width, const float height, const float dist1, const float dist2)
//{
//	std::vector<cv::Point2f> rps(4), ips(4);
//	if( road2image.empty() )
//		return STEREO_ERR_NOSRC;
//	int res = STEREO_SUCCEED;
//
//	rps[0] = cv::Point2f(-width, dist1);
//	rps[1] = cv::Point2f(+width, dist1);
//	rps[2] = cv::Point2f(+width*2.0, dist2);
//	rps[3] = cv::Point2f(-width*2.0, dist2);
//	cv::perspectiveTransform(rps, ips, road2image);
//
//	float top = MIN(ips[3].y, ips[2].y), btm = MAX(ips[0].y, ips[1].y);
//	float max_height = height <= 0 ? imageSize.height - 20 :
//			MIN(imageSize.height - 20, height);
//	LOGI("top: %f, btm: %f, max_height: %f.", top, btm, max_height);
//
//	if( btm > max_height )
//		btm = max_height;
//	if( top < imageSize.height/2 - 40 )
//		top = imageSize.height/2 - 40;
//
//	if( top > max_height || btm < imageSize.height/2 - 40 || top >= btm )
//	{
//		res = STEREO_ERR_PARAM;
//		btm = max_height;
//		top = imageSize.height/2 - 40;
//	}
//
//	rect = cv::Rect(0, top, imageSize.width, btm - top);
//	line_tables.resize(rect.height);
//	return res;
//}

//	std::vector<cv::Point> mps(4);
//
//	mps[0] = cv::Point(0, btm);
//	mps[1] = cv::Point(imageSize.width, btm);
//	mps[2] = cv::Point(imageSize.width*.7, top);
//	mps[3] = cv::Point(imageSize.width*.3, top);

// #undef R2Y
// #undef G2Y
// #undef B2Y

// enum
// {
//     yuv_shift = 14,
//     xyz_shift = 12,
//     R2Y = 4899,
//     G2Y = 9617,
//     B2Y = 1868,
//     BLOCK_SIZE = 256
// };

// const int ITUR_BT_601_CY  = 1220542;
// const int ITUR_BT_601_CUB = 2116026;
// const int ITUR_BT_601_CUG = -409993;
// const int ITUR_BT_601_CVG = -852492;
// const int ITUR_BT_601_CVR = 1673527;
// const int ITUR_BT_601_SHIFT = 20;

// #define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)


// struct toYellowGrayInvokerByR : cv::ParallelLoopBody
// {
//     const uchar * src_data;
//     uchar * dst_data;
//     const size_t src_step, dst_step;
//     const size_t width, height, rowStep;

//     toYellowGrayInvokerByR(uchar *_dst_data, const size_t _dst_step,
//     		const uchar *_src_data, const size_t _src_step,
//     		const size_t _width, const size_t _height, const size_t _rowStep):
//     			src_data(_src_data), dst_data(_dst_data), src_step(_src_step),
//     			dst_step(_dst_step), width(_width), height(_height), rowStep(_rowStep)
// 	{}

//     uchar toYellowScale(uchar gray, uchar r, uchar g, uchar b) const
//     {
//     	return cv::saturate_cast<uchar>(r > gray ? r*2 - gray: r);
//     }

//     uchar toYellowScale(uchar g, uchar r) const
//     {
//     	return cv::saturate_cast<uchar>( (g + r + r + r) >> 2 );
//     }

//     uchar toYellowScale(int g, int r) const
//     {
//     	return cv::saturate_cast<uchar>( (g + r + r + r) >> 2 );
//     }
// };


// /// template<int bIdx, int uIdx, int yIdx> /// YUYV bIdx=0, uIdx=0, yIdx=0
// struct YUV422toRGB888Invoker : cv::ParallelLoopBody
// {
//     const uchar * src_data;
//     uchar * dst_data;
//     const size_t src_step, dst_step;
//     const size_t width, height;

//     YUV422toRGB888Invoker(uchar *_dst_data, const size_t _dst_step,
//     		const uchar *_src_data, const size_t _src_step,
//     		const size_t _width, const size_t _height):
//     			src_data(_src_data), dst_data(_dst_data), src_step(_src_step),
//     			dst_step(_dst_step), width(_width), height(_height)
// 	{}

//     void operator()(const cv::Range& range) const
//     {
//         int rangeBegin = range.start;
//         int rangeEnd = range.end, sstep1 = src_step, sstep2 = src_step/2;

//         int ustart = height * src_step, vstart = ustart + height * src_step/2;
//         const uchar* ysrc = src_data + rangeBegin * sstep1;
//         const uchar* usrc = src_data + ustart + rangeBegin * sstep2;
//         const uchar* vsrc = src_data + vstart + rangeBegin * sstep2;

//         for (int j = rangeBegin; j < rangeEnd; j++ )
//         {
//             uchar* row = dst_data + dst_step * j;

//             for( size_t i = 0; i < width; i += 2, row += 6 )
//             {
//                 int u = int(usrc[i>>1]) - 128;
//                 int v = int(vsrc[i>>1]) - 128;

//                 int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
//                 int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v+ ITUR_BT_601_CUG * u;
//                 int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

//                 int y00 = MAX(0, int(ysrc[i+0]) - 16) * ITUR_BT_601_CY;
//                 int y01 = MAX(0, int(ysrc[i+1]) - 16) * ITUR_BT_601_CY;

//                 row[2] = cv::saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
//                 row[1] = cv::saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
//                 row[0] = cv::saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

//                 row[5] = cv::saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
//                 row[4] = cv::saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
//                 row[3] = cv::saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
//             }

// 			ysrc += sstep1; usrc += sstep2; vsrc += sstep2;
//         }
//     }
// };


// struct BGR2YellowGrayInvoker : toYellowGrayInvokerByR
// {
// 	BGR2YellowGrayInvoker(uchar *_dst_data, const size_t _dst_step,
// 			const uchar *_src_data, const size_t _src_step,
// 			const size_t _width, const size_t _height, const size_t _rowStep = 1):
// 				toYellowGrayInvokerByR(_dst_data, _dst_step, _src_data, _src_step,
// 						_width, _height, _rowStep)
//     {
//         const int coeffs0[] = { R2Y, G2Y, B2Y };

//         int b = 0, g = 0, r = (1 << (yuv_shift-1));
//         int db = coeffs0[2], dg = coeffs0[1], dr = coeffs0[0];

//         for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
//         {
//             tab[i] = b;
//             tab[i+256] = g;
//             tab[i+512] = r;
//         }
//     }

//     void operator()(const cv::Range& range) const
//     {
//         const int rangeBegin = (range.start/rowStep + (range.start%rowStep?1: 0))*rowStep;
//         const int rangeEnd = (range.end/rowStep + (range.end%rowStep?1: 0))*rowStep;

//         for (int row = rangeBegin; row < rangeEnd; row += rowStep )
//         {
//             const uchar *src = src_data + src_step * row;
//             uchar* dst = dst_data + dst_step * row;

//             for( size_t i = 0; i < width; i++, dst++, src += 3 )
//             {
//             	*dst = toYellowScale(src[1], src[2]);
//             }
//         }
//     }

//     int tab[256*3];
// };


// struct YUV422toYellowGrayInvoker : toYellowGrayInvokerByR
// {
//     YUV422toYellowGrayInvoker(uchar *_dst_data, const size_t _dst_step,
//     		const uchar *_src_data, const size_t _src_step,
//     		const size_t _width, const size_t _height, const size_t _rowStep = 1):
// 				toYellowGrayInvokerByR(_dst_data, _dst_step, _src_data, _src_step,
// 						_width, _height, _rowStep)
//     {}

//     void operator()(const cv::Range& range) const
//     {
//         const int rangeBegin = (range.start/rowStep + (range.start%rowStep?1: 0))*rowStep;
//         const int rangeEnd = (range.end/rowStep + (range.end%rowStep?1: 0))*rowStep;

//         const int ustart = height * src_step, vstart = ustart + height * src_step/2;
//         const int sstep1 = src_step, sstep2 = src_step/2;

//         const uchar* ysrc = src_data + rangeBegin * sstep1;
//         const uchar* usrc = src_data + ustart + rangeBegin* sstep2;
//         const uchar* vsrc = src_data + vstart + rangeBegin* sstep2;

//         for (int row = rangeBegin; row < rangeEnd; row += rowStep )
//         {
//             uchar* rowData = dst_data + dst_step * row;

//             for( size_t i = 0; i < width; i += 2, rowData += 2 )
//             {
//                 int y00 = MAX(0, int(ysrc[i+0]) - 16) * ITUR_BT_601_CY;
//                 int y01 = MAX(0, int(ysrc[i+1]) - 16) * ITUR_BT_601_CY;

//                 int u = int(usrc[i>>1]) - 128;
//                 int v = int(vsrc[i>>1]) - 128;

//                 int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
//                 int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v+ ITUR_BT_601_CUG * u;

//                 int r0 = (y00 + ruv) >> ITUR_BT_601_SHIFT;
//                 int g0 = (y00 + guv) >> ITUR_BT_601_SHIFT;

//                 int r1 = (y01 + ruv) >> ITUR_BT_601_SHIFT;
//                 int g1 = (y01 + guv) >> ITUR_BT_601_SHIFT;

//                 rowData[0] = toYellowScale(g0, r0);
//                 rowData[1] = toYellowScale(g1, r1);
//             }

// 			ysrc += sstep1; vsrc += sstep2; usrc += sstep2;
//         }
//     }
// };


// inline void cvtYUV422toYellowGray(uchar *dst_data, const size_t dst_step,
// 		const uchar *src_data, const size_t src_step,
// 		const size_t width, const size_t height, const size_t rowStep)
// {
// 	YUV422toYellowGrayInvoker converter(dst_data, dst_step, src_data,
// 			src_step, width, height, rowStep);
//     if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
//         parallel_for_(cv::Range(0, height), converter);
//     else
//         converter(cv::Range(0, height));
// }

// cv::Mat cvtYUV422toYellowGray(const uchar* src_data, const cv::Size size,
// 		const size_t rowStep)
// {
// 	cv::Mat gray(size, CV_8UC1, cv::Scalar::all(0));
// 	cvtYUV422toYellowGray(gray.data, size.width, src_data,
// 			size.width, size.width, size.height, rowStep);
// 	return gray;
// }

// cv::Mat cvtYUV422toYellowGray(uchar *src_data, const cv::Size size,
// 		const cv::Rect rect, const size_t rowStep)
// {
// 	int _ysrccnt = size.width*size.height, _usrccnt = _ysrccnt/2;
// 	int _ydstcnt = rect.width*rect.height, _udstcnt = _ydstcnt/2;
// 	cv::Rect uvrect(rect.x/2, rect.y, rect.width/2, rect.height);
// 	uchar _src_data[rect.width*rect.height*2];

// 	const cv::Mat ysrc(size.height, size.width  , CV_8UC1, src_data);
// 	const cv::Mat usrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt);
// 	const cv::Mat vsrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt + _usrccnt);

// 	ysrc(  rect).copyTo(cv::Mat(rect.height, rect.width  , CV_8UC1, _src_data));
// 	usrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt));
// 	vsrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt + _udstcnt));

// 	return cvtYUV422toYellowGray(_src_data, rect.size(), rowStep);
// }


// inline void cvtBGR2YellowGray(uchar *dst_data, const size_t dst_dataStep,
// 		const uchar *src_data, const size_t src_dataStep,
// 		const size_t width, const size_t height, const size_t rowStep)
// {
// 	BGR2YellowGrayInvoker converter(dst_data, dst_dataStep,
// 			src_data, src_dataStep, width, height, rowStep);
//     if( width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION )
//     	parallel_for_(cv::Range(0, height), converter);
//     else
//         converter(cv::Range(0, height));
// }

// cv::Mat cvtBGR2YeloowGray(const cv::Mat src)
// {
// 	cv::Mat gray(src.size(), CV_8UC1, cv::Scalar::all(0));
// 	cvtBGR2YellowGray(gray.data, gray.step, (uchar*)src.ptr<cv::Vec3b>(0),
// 			src.step, src.cols, src.rows, 1);
// 	return gray;
// }

// cv::Mat cvtBGR2YeloowGray(const cv::Mat src, const cv::Rect rect, const int rowStep)
// {
// 	cv::Mat gray(rect.size(), CV_8UC1, cv::Scalar::all(0));
// 	cvtBGR2YellowGray(gray.data, gray.step, (uchar*)src.ptr<cv::Vec3b>(0)
// 			+ src.step * rect.y + rect.x*3, src.step, rect.width, rect.height, rowStep);
// 	return gray;
// }


// inline void cvtYUV422toRGB(uchar *dst_data, size_t dst_step, const uchar *src_data, size_t src_step,
//                            int width, int height)
// {
//     YUV422toRGB888Invoker converter(dst_data, dst_step, src_data, src_step, width, height);
//     if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
//         parallel_for_(cv::Range(0, height), converter);
//     else
//         converter(cv::Range(0, height));
// }


// cv::Mat cvtYUV422toBGR(const uchar* src_data, const cv::Size size)
// {
// 	cv::Mat gray(size, CV_8UC3, cv::Scalar::all(0));
// 	cvtYUV422toRGB(gray.data, gray.step, src_data, size.width, size.width, size.height);
// 	return gray;
// }

// cv::Mat cvtYUV422toBGR(uchar *src_data, const cv::Size size, const cv::Rect rect)
// {
// 	int _ysrccnt = size.width*size.height, _usrccnt = _ysrccnt/2;
// 	int _ydstcnt = rect.width*rect.height, _udstcnt = _ydstcnt/2;
// 	cv::Rect uvrect(rect.x/2, rect.y, rect.width/2, rect.height);
// 	uchar _src_data[rect.width*rect.height*2];

// 	const cv::Mat ysrc(size.height, size.width  , CV_8UC1, src_data);
// 	const cv::Mat usrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt);
// 	const cv::Mat vsrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt + _usrccnt);

// 	ysrc(  rect).copyTo(cv::Mat(rect.height, rect.width  , CV_8UC1, _src_data));
// 	usrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt));
// 	vsrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt + _udstcnt));

// 	return cvtYUV422toBGR(_src_data, rect.size());
// }


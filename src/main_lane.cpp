#include <iostream>
#include <stdio.h>
#include <getopt.h>

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "core/utils.hpp"
#include "core/plot.hpp"
#include "core/comimg.hpp"
#include "core/stereo.hpp"

#include "ground/distgrid.hpp"

#include "yuvcvt.hpp"
#include "track.hpp"
#include "locate.hpp"

//#define TEST_TIME                 1


static struct CmdLine
{
	int save, debug, skip, jump, tri, ep, car_width;
	float pixel_unit, shift;
	char path[128], input[128];
	FileSearch search;

	CmdLine(): save(0), debug(0), skip(0), jump(-1), tri(0), ep(0), car_width(2000),
			pixel_unit(-1), shift(250), search("image", "jpg")
	{
		path[0] = input[0] = 0;
	}
}line;

static void print_help();
static int ParseCmdLine(CmdLine &line, int argc, char* argv[]);

static int TrackLanes(void* const userData, const cv::Mat &image,
		const std::string file);


struct LaneTrackArg
{
	const StereoCamera *const camera;
	const BVTran *tran;
	const cv::Size blockSize;

	cv::Ptr<LaneBlockDetect> detect;
	cv::Ptr<LaneTrack> track;

	DistGrid grid;
	cv::Mat gray, datas, epts;

	int left_count, rght_count;
	double m0, mx, my;
	cv::TickMeter tick;
	LaneLocate locate;

	LaneTrackArg(const StereoRectify *const rectify, const Birdview *bird):
		camera(rectify->camera), tran(0), blockSize(4, 3),
		grid(10, 5, 500, 100, 500, 0), datas(3, 500, CV_64FC1), epts(rectify->size, CV_32FC1),
		left_count(0), rght_count(0), m0(0), mx(0), my(0), locate(camera->imageSize, bird->ori_left.road2image)
	{
		track.release();
		detect = LaneBlockDetect::CreateDetect(EDGE_GRADIENT_THRESH*12, 20);// *12, *16,

		if( bird != 0 && bird->canTran && !bird->ori_left.road2image.empty() )
		{
			cv::Mat road2im, im2road;

//			tran = line.tri ? &((TriBV*)bird)->ori_record: &bird->ori_left;
			tran = &bird->ori_left;
			tran->image2road.copyTo(im2road);
			tran->road2image.copyTo(road2im);

#if ADASLANE_HALF_IMAGE
			road2im.rowRange(0, 2) *=.5;
			im2road.colRange(0, 2) *= 2;
			detect->SetDetectRegion(camera->imageSize/2, road2im, 350, 5000);
#else
			detect->SetDetectRegion(camera->imageSize, road2im, 350, 5000);
#endif
			track = cv::makePtr<LaneTrackBasedRoad>(detect, tran->road2image,
					im2road, line.car_width *.1);
			LOGI("lane detecting uses road-based model.");
		}
		else
		{
			const int h = camera->imageSize.height;
			detect->SetDetectRegion(camera->imageSize/2, h-20, h/2);
			track = cv::makePtr<LaneTrackOpenModel>(detect);
			LOGI("##Warning: road parameter has wrong, lane detecting uses open model.");
		}

		std::cout<<"DetectRect: "<<detect->GetDetectRect()<<std::endl;

		epts = 0; datas = 0;
		grid.GenGroundPts();
		grid.CalcImagePts(bird);
	}

	int operator()(const cv::Mat image)
	{
		track->Forward();
		if( track->NeedDetected() )
		{
#if ADASLANE_HALF_IMAGE
			cv::Mat temp = image;

			cv::resize(image, temp, cv::Size(), .5, .5);
			gray = cvtBGR2YeloowGray(temp, track->rect, detect->GetRowStep()->stripe);

			tick.start();

			(*track)(gray, detect, blockSize);

			if( track->left.index == track->frameIndex )
			{
				track->left.image_line[2] *= 2;
				track->left.image_line[3] *= 2;
			}
			if( track->rght.index == track->frameIndex )
			{
				track->rght.image_line[2] *= 2;
				track->rght.image_line[3] *= 2;
			}
#else
			//gray = cvtBGR2YeloowGray(image, track->rect, detect->GetRowStep()->stripe);

			tick.start();
			cv::cvtColor(image(track->rect), gray, cv::COLOR_BGR2GRAY);
			(*track)(gray, detect, blockSize);
#endif

			// if( track->left.index == track->frameIndex )
			// 	locate(track->left.road_line, image, 40);

			if( track->left.index == track->frameIndex || track->rght.index == track->frameIndex )
				locate(track->left.road_line, track->rght.road_line, image, 40);

			track->FrushStep(1, 2, 3);
			tick.stop();

//			LaneBlock *left_block = track->blocks()[track->leftIndex];
//			LaneBlock *rght_block = track->blocks()[track->rghtIndex];

			if( track->left.index == track->frameIndex )
				left_count++;
			if( track->rght.index == track->frameIndex )
				rght_count++;

			cv::Point2f rept;
			if( line.ep && track->CalcRectifyEndPoint(rept, camera->compact) )
			{
				epts.at<float>(rept)++;
				m0++; mx += rept.x; my += rept.y;
			}
		}

//		PlotDiffAngle();
//		PlotShift();

		cv::Mat dispImg = image.clone(), temp = line.tri ? dispImg: dispImg(camera->left);
		detect->Show(temp);
		track->Show(temp);
//		cv::circle(temp, ep, 5, cv::Scalar(0, 0, 255), 2);

		if( line.pixel_unit > 0 && tran != 0 && !tran->image2road.empty() )
		{
			cv::Mat temp = image(camera->left).clone(), bird;
			LaneTrackBasedRoad *trackroad = dynamic_cast<LaneTrackBasedRoad*>(track.get());

			if( (*tran)(temp, bird) == 0 && trackroad != 0 )
			{
				trackroad->ShowRoadLines(bird, line.pixel_unit, cv::Point2f(0, line.shift));
				dispImg = ConcatImages(dispImg(camera->left), bird);
			}
		}
		else if( line.pixel_unit <= 0 )
		{
			const StereoRectify *rectify = camera->compact;
			if( rectify->size.height > camera->imageSize.height )
			{
				cv::Mat temp(rectify->size.height, camera->imageSize.width*2, dispImg.type(), cv::Scalar(255, 255, 255));
				dispImg.copyTo(temp.rowRange(0, camera->imageSize.height));
				dispImg = temp;
			}

			const int dh = MAX(0, (camera->imageSize.height - rectify->size.height)/2);
			const int dw = MAX(0, (camera->imageSize.width  - rectify->size.width )/2);
			const cv::Rect rect(0, 0, camera->imageSize.width*2, camera->imageSize.height);
			cv::Mat rdisp = dispImg(rectify->left + cv::Point(camera->imageSize.width + dw, dh));

			dispImg(camera->rght) = 0;
			(*rectify)(image(rect), rdisp, StereoRectify::LEFT);

			if( line.ep )
				grid.DrawGridPts(rdisp, grid.lips);

			float lines[6];
			memset(lines,0, sizeof(lines));
			track->CalcRectifyLine(rectify, lines);

			if( lines[2] > 0 )
			{
				cv::Point p1(lines[1], 0), p2(lines[1]+ lines[0]*rectify->size.height,
						rectify->size.height);
				cv::line(rdisp, p1, p2, cv::Scalar(255, 0, 0), 2);
			}
			if( lines[5] > 0 )
			{
				cv::Point p1(lines[4], 0), p2(lines[4]+ lines[3]*rectify->size.height,
						rectify->size.height);
				cv::line(rdisp, p1, p2, cv::Scalar(255, 0, 0), 2);
			}
		}

#if !ANDROID && !TEST_TIME
		if( line.ep && track->frameIndex %100 == 0 && m0 > 0 )
		{
			cv::Mat temp;
			cv::normalize(epts, temp, 0, 255, cv::NORM_MINMAX);
			temp.convertTo(temp, CV_8U);
			cv::imshow("end points", temp);
		}
		cv::imshow("dispImg", dispImg);
#endif

		if( track->frameIndex == line.jump )
			line.debug = 1;
		return STEREO_SUCCEED;
	}

	void PlotDiffAngle()
	{
		if( track->leftIndex < 0 || track->rghtIndex < 0 )
			return;
		const std::vector<LaneBlock*> blocks = detect->GetLaneBlocks();
		LaneBlock *left_block = blocks[track->leftIndex];
		LaneBlock *rght_block = blocks[track->rghtIndex];

		const int last = datas.cols - 1;
		double *lad = datas.ptr<double>(0);
		double *rad = datas.ptr<double>(1);
		double *dad = datas.ptr<double>(2);

		memcpy(&lad[0], &lad[1], sizeof(lad[0])*last);
		memcpy(&rad[0], &rad[1], sizeof(lad[0])*last);
		memcpy(&dad[0], &dad[1], sizeof(lad[0])*last);

		lad[last] = left_block->angle * 180/M_PI - 90;
		rad[last] = rght_block->angle * 180/M_PI - 90;
		dad[last] = lad[last] - rad[last];

		cv::Mat dispImg;
		cv::Ptr<utils::Plot2d> plot = utils::createPlot2d(datas);

		plot->setPlotSize(1240, 440);
//		plot->setMaxY(420);
		plot->render(dispImg, utils::PLOT_LINE);
		cv::imshow("plot diff angle", dispImg);

		if( line.save && track->frameIndex%datas.cols == 0 )
		{
			char fname[256];
			memset(fname, 0, 256);
			sprintf(fname, "%s/%s_%02d.jpg", line.path, line.input,
					track->frameIndex/datas.cols);
			cv::imwrite(fname, dispImg);
		}
	}

	void PlotShift()
	{
		LaneTrackBasedRoad* trackroad = dynamic_cast<LaneTrackBasedRoad*>(track.get());
		if( trackroad == NULL )
			return;

		const int last = datas.cols - 1;
		double *shift = datas.ptr<double>(1);
		double *width = datas.ptr<double>(0);

		memcpy(&shift[0], &shift[1], sizeof(double) * last);
		memcpy(&width[0], &width[1], sizeof(double) * last);

		shift[last] = trackroad->GetLaneShift();
		width[last] = trackroad->GetLaneWidth();
//		shift[last] = fabs(trackroad->GetLaneShift());
//		width[last] = fabs(trackroad->GetLaneWidth());

		cv::Mat data = datas, dispImg;
		cv::Ptr<utils::Plot2d> plot = utils::createPlot2d(data);

		plot->setPlotSize(1240, 440);
		plot->setMaxY(420);
		plot->render(dispImg, utils::PLOT_LINE);

		cv::imshow("plot shift", dispImg);
		if( line.save && track->frameIndex%datas.cols == 0 )
		{
			char fname[512];
			memset(fname, 0, 256);
			sprintf(fname, "%s/%s_%02d.jpg", line.path, line.input,
					track->frameIndex/data.cols);
			cv::imwrite(fname, dispImg);
		}
	}

	void DrawLaneLine(cv::Mat image, const int flag, const float k, const float b,
			const int width = 2)
	{
		const cv::Scalar red(0, 0, 255), blue(255, 0, 0), green(0, 255, 0);

		if( flag&LaneTrack::DETECTED_LINE )
		{
			const cv::Scalar color = flag&LaneTrack::RUNIN_LINE ? red : blue;
			cv::line(image, cv::Point(b, 0), cv::Point(k*image.rows+b, image.rows),
					color, width);
			if( flag&LaneTrack::CONFIRM_LINE )
				DrawDottedCircle(image, k, b, blue, 5, 2, 0);
		}
		else if( flag&LaneTrack::CALCULATED_LINE )
		{
			const cv::Scalar color = flag&LaneTrack::RUNIN_LINE ? red : green;
			cv::line(image, cv::Point(b, 0), cv::Point(k*image.rows+b, image.rows),
					color, width);
			if( flag&LaneTrack::CONFIRM_LINE )
				DrawDottedCircle(image, k, b, green, 5, 2, 0);
		}
	}
};


#if TEST_TIME
void TestTime_CvtColor(const std::vector<cv::Mat> &images);

void TestTime_ScanEdgeLines(const std::vector<cv::Mat> &images,
		LaneBlockDetect *detect, DetectParam *rowStep);

static void TestTime_rectify(const std::vector<cv::Mat> &images, StereoCamera *camera);

static void TestTime_warpPerspective(const std::vector<cv::Mat> &images, Birdview *bird, StereoCamera *camera);

static void TestTime_remap(const std::vector<cv::Mat> &images,
		Birdview *bird, StereoCamera *camera);

void CvtYUV422toRGB888(uchar *ysrc_data, uchar *usrc_data, uchar *vsrc_data,
		const size_t src_step, uchar *dst_data, const size_t img_width, const size_t img_height);

void CvtYUV422toYellowGray(uchar *ysrc_data, uchar *usrc_data, uchar *vsrc_data,
		const size_t src_step, const size_t src_x, const size_t src_y,
		uchar *dst_data, const size_t img_width, const size_t img_height);
#endif


int main(int argc, char* argv[])
{
	LOGI("%s", LaneTrack::GetVersion());

	int res = ParseCmdLine(line, argc, argv);
	if( res < 0 )
		return STEREO_ERR_ARG;

	cv::Ptr<StereoCamera> camera = cv::makePtr<StereoCamera>();
	cv::Ptr<Birdview> road = cv::makePtr<StereoBV>(camera->compact);

//	if( !line.tri )
//	{
//		camera = cv::makePtr<StereoCamera>();
//		road = cv::makePtr<StereoBV>(camera->compact);
//	}
//	else
//	{
//		camera = cv::makePtr<TriCamera>(cv::Size(640, 480));
//		road = cv::makePtr<TriBV>(camera->compact);
//	}

	if( camera->Open(line.path) == 0 )
	{
		res = camera->GenUndistortMap();
		if( !camera->isRectified() || res != STEREO_SUCCEED )
		{
			LOGI("##Error: camera can't be rectified.");
			return STEREO_ERR_NOSRC;
		}

		road->Open(line.path);
		if( road->canTran && line.pixel_unit > 0 )
			road->genMapOrigin(line.pixel_unit, cv::Point2f(0, line.shift));
	}

#if !TEST_TIME

	const cv::Rect rect(0, 0, 1280, 480);
	LaneTrackArg trackArg(camera->compact, road);

	if( strlen(line.input) > 0 )
	{
		char fname[256];
		cv::Mat frame, image;
		cv::VideoCapture cap;

		sprintf(fname, "%s/%s", line.path, line.input);
		cap.open(fname);

		trackArg.track->frameIndex = line.skip;
		cap.set(cv::CAP_PROP_POS_FRAMES, trackArg.track->frameIndex);
		if( line.tri )
		{
			cap.set(cv::CAP_PROP_FRAME_WIDTH , 640);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		}

		if( !cap.isOpened() )
			return STEREO_ERR_ARG;

		while( cap.grab() )
		{
			if( !cap.retrieve(frame) || frame.empty() )
				break;

			if( line.tri )
				cv::resize(frame.colRange(160, 960+160), image,
						camera->imageSize);
			else
				image = frame(rect);

			res = trackArg(image);
			if( res != 0 )
				break;

#if !ANDROID && !TEST_TIME
			int ch = cv::waitKey(line.debug ? 0: 5);
			if( ch == 'd' || ch == 'D' )
				line.debug = !line.debug;
			if( ch == 'S' || ch == 's' )
			{
				sprintf(fname, "%s/image_%04d.jpg", line.path, trackArg.track->frameIndex);
				cv::imwrite(fname, image);
			}
			else if( ch == 'B' || ch == 'b' )
			{
				trackArg.track->frameIndex -=200;
				cap.set( cv::CAP_PROP_POS_FRAMES, trackArg.track->frameIndex );
			}
			else if( ch == 27 )
				break;
#endif
		}

#if CALC_CURVE_LANE
		cv::Mat curs = trackArg.curvatures * 180/M_PI;
		utils::createPlot2d(curs)->render(image, utils::PLOT_POINT);
		cv::imshow("plot curvatures", image);
		cv::waitKey(0);
#endif

		cap.release();
		LOGI("frames %d, detected %d times, left line %d, right line %d",
				trackArg.track->frameIndex, trackArg.track->GetDetectCnt(),
				trackArg.left_count, trackArg.rght_count);
		LOGI("VISUAL: use time: %f second, image count: %ld.", trackArg.tick.getTimeSec(), (long)trackArg.tick.getCounter());

		if( line.ep )
		{
			cv::Point maxLoc(trackArg.mx/trackArg.m0, trackArg.my/trackArg.m0);
			cv::Rect validRect(maxLoc.x - 5, maxLoc.y - 5, 11, 11);
			cv::Moments m = cv::moments (trackArg.epts(validRect));
			LOGI("max location: [%d, %d]", maxLoc.x, maxLoc.y);
			LOGI("end point: [%f, %f]", m.m10/m.m00+validRect.x, m.m01/m.m00+validRect.y);
			LOGI("end point: [%f, %f]", trackArg.mx/trackArg.m0, trackArg.my/trackArg.m0);
		}
	}
	else
	{
		BiImage combimg(line.path, &line.search);
		combimg.Search(true);
		combimg.load(TrackLanes, (void*)&trackArg);
	}

#else // TEST_TIME

	if( strlen(line.input) )
	{
		printf("%s in %s(%d).\n", __func__, __FILE__, __LINE__);

		cv::Mat yuvimg, bgrimg, gray, temp;
		cv::Mat dsts[] = {gray, temp};
	    int frmto[] = {0,0, 1,1, 2,2, 3,3};

		res = load_image(yuvimg, line.path, line.input, 1, 0);
		cv::imshow("yuv image", yuvimg);

		int src_step = yuvimg.cols, height = yuvimg.rows>>1;
        int ustart = height * src_step, vstart = ustart + height * src_step/2;
        uchar* src_data = yuvimg.data;

        gray = cv::Mat(300, 400, CV_8UC1);
		CvtYUV422toYellowGray(src_data, src_data + ustart, src_data + vstart,
				src_step, 100, 100, gray.data, 400, 300);
		cv::imshow("gray image", gray);

        bgrimg = cv::Mat(height, src_step, CV_8UC3);
        CvtYUV422toRGB888(src_data, src_data + ustart, src_data + vstart, src_step,
        		bgrimg.data, src_step, height);
        cv::cvtColor(gray, bgrimg(cv::Rect(100, 100, 400, 300)), cv::COLOR_GRAY2BGR);
		cv::imshow("bgr image", bgrimg);

		cv::imwrite("image_bgr.jpg", bgrimg);

		cv::waitKey(0);
	}
	else
	{
		BiImage combimg(line.path, &line.search);
		combimg.Search(true);

		if( !combimg.HasImages() )
			return 0;
		cv::Mat image, _inte;

		const int len = combimg.left_size();
		std::vector<cv::Mat> clrImgs(len), grayImgs(len);
		const std::vector<std::string> &files = combimg.GetFileList1();

		for( int i = 0; i < len; i++ )
		{
//				load_image(images[i], line.path, files[i].c_str());
			load_image(image, line.path, files[i].c_str());
			clrImgs[i] = image;
			color2gray(image, grayImgs[i]);
		}

		DetectParam rowStep(1);
		cv::Ptr<LaneBlockDetect> detect = LaneBlockDetect::CreateDetect(&rowStep, 4, 1, -2, 20);
		detect->SetDetectRegion(camera->imageSize, road->ori_left.road2image, 350, 0, 350, 3000);
		TestTime_ScanEdgeLines(grayImgs, detect, &rowStep);
	}

#endif //TEST_TIME

	return res;
}


//			cv::Vec4f ol(0.831549, 0.555451, 480.417, 372.967), rl;
//			std::cout<<ol<<std::endl;
//			TransLine(ol, rl, tran->image2road, 20);
//			std::cout<<rl<<std::endl;
//
//			ol = cv::Vec4f(0.831908, 0.554913, 245.944, 190.222);
//			std::cout<<ol<<std::endl;
//			TransLine(ol, rl, im2road, 10);
//			std::cout<<rl<<std::endl;

//		if( bird != 0 && bird->canTran && !bird->ori_left.road2image.empty() )
//		{
//			tran = line.tri ? &((TriBV*)bird)->ori_record: &bird->ori_left;
//			res = detect->SetDetectRegion(camera->imageSize, tran->road2image, 350, 0,
//					350, 5000);
//			rectify->getOriginPoints(&bird->end_point, &ep, 1, &camera->cam_left);
//		}
//		else
//		{
//			const int h = camera->imageSize.height;
//			res = STEREO_ERR_NOSRC;
//			detect->SetDetectRegion(camera->imageSize, h-20, h/2);
//		}
//
//		if( res != STEREO_SUCCEED )
//		{
//			LOGI("##Warning: road parameter has wrong, lane detecting uses open model.");
//			track = cv::makePtr<LaneTrackOpenModel>(detect);
//		}
//		else
//		{
//			LOGI("lane detecting uses road-based model.");
//			track = cv::makePtr<LaneTrackBasedRoad>(detect, tran->road2image,
//					tran->image2road, line.car_width *.1);
//		}

//			grid.GenGroundPts();
//			std::cout<<grid.rps.size()<<std::endl;
//
//			cv::perspectiveTransform(grid.rps, grid.lips, tran->road2image);
//			std::cout<<grid.rps<<std::endl;
//			std::cout<<grid.lips<<std::endl;
//			cv::perspectiveTransform(grid.rps, grid.lips, road2im);
//			std::cout<<grid.lips<<std::endl;

//			rectify->getOriginPoints(&bird->end_point, &ep, 1, &camera->cam_left);

#if TEST_TIME
void TestTime_CvtColor(const std::vector<cv::Mat> &images)
{
	const int len = images.size();

	long long int st = cv::getTickCount();
	LOGI("start time: %lld", st);

	for( int j = 0; j < 100; j++ )
	{
		for( int i = 0; i < len; i++ )
			cvtBGR2YeloowGray(images[i]);
	}

	long long int ed = cv::getTickCount();
	LOGI("end time: %lld", ed);
	double alltime = (double)(ed - st)/ cv::getTickFrequency();
	LOGI("use time: %f sec, image count: %d", alltime, len);
}

void TestTime_ScanEdgeLines(const std::vector<cv::Mat> &images,
		LaneBlockDetect *detect, DetectParam *rowStep)
{
	const int len = images.size();

	const cv::Rect rect = detect->GetDetectRect();
	char fname[256];

	cv::Mat inte = cv::Mat::zeros(rect.height+1, rect.width+1, CV_32SC1);
	cv::Mat edge = cv::Mat::zeros(rect.height, rect.width, CV_32SC1);

	long long int st = cv::getTickCount();
	LOGI("start time: %lld", st);

	for( int j = 0; j < 100; j++ )
	{
		for( int i = 0; i < len; i++ )
			(*detect->scan)(images[i](rect), inte, edge, rowStep->stripe);
//			detect->Clear();
//			detect->ScanEdgeLines(images[i](rect));
	}

	long long int ed = cv::getTickCount();
	LOGI("end time: %lld", ed);
	double alltime = (double)(ed - st)/ cv::getTickFrequency();
	LOGI("use time: %f sec, image count: %d", alltime, len);
}

static void TestTime_rectify(const std::vector<cv::Mat> &images, StereoCamera *camera)
{
	const int len = images.size();
	long long int st = cv::getTickCount();
	LOGI("start time: %lld", st);
	cv::Mat diff, img;

	for( int i = 0; i < len; i++ )
	{
		for( int j = 0; j < 200; j++ )
		{
			(*camera->compact)(images[i], img, StereoRectify::ALL, 0);
		}
	}

	long long int ed = cv::getTickCount();
	LOGI("end time: %lld", ed);
	double alltime = (double)(ed - st)/ cv::getTickFrequency();
	LOGI("use time: %f sec, image count: %d", alltime, len);
}

static void TestTime_warpPerspective(const std::vector<cv::Mat> &images, Birdview *bird, StereoCamera *camera)
{
	const int len = images.size();
	long long int st = cv::getTickCount();
	LOGI("start time: %lld", st);
	cv::Mat diffimg;

	for( int i = 0; i < len; i++ )
	{
		for( int j = 0; j < 200; j++ )
		{
			bird->genRight2Left(images[i](camera->rght), diffimg, 1,
					camera->imageSize, cv::INTER_NEAREST);
		}
	}

	long long int ed = cv::getTickCount();
	LOGI("end time: %lld", ed);
	double alltime = (double)(ed - st)/ cv::getTickFrequency();
	LOGI("use time: %f sec, image count: %d", alltime, len);
}

void GenMap(const cv::Mat tran, cv::Mat &map, const cv::Size size)
{
	std::vector<cv::Point2f> lips, rips;

	for( int i = 0; i < size.height; i++ )
	{
		for( int j = 0; j < size.width; j++ )
		{
			lips.push_back(cv::Point2f(j, i));
		}
	}

	cv::perspectiveTransform(lips, rips, tran);
	map = cv::Mat::zeros(size, CV_32FC2);

	for( int i = 0; i < size.height; i++ )
	{
		cv::Point2f *pd = map.ptr<cv::Point2f>(i);
		cv::Point2f *ps = &rips[i*size.width];

		for( int j = 0; j < size.width; j++ )
		{
			pd[j] = ps[j];
		}
	}
}

static void TestTime_remap(const std::vector<cv::Mat> &images,
		Birdview *bird, StereoCamera *camera)
{
	cv::Mat tran = bird->ori_rght.road2image*bird->ori_left.image2road;
	cv::Mat diffimg, map;//(camera->imageSize, CV_32FC2), diffimg;

	GenMap(tran, map, camera->imageSize);

	const int len = images.size();
	long long int st = cv::getTickCount();
	LOGI("start time: %lld", st);

	for( int i = 0; i < len; i++ )
	{
		for( int j = 0; j < 200; j++ )
		{
			cv::remap(images[i](camera->rght), diffimg, map, cv::noArray(), cv::INTER_NEAREST);
		}
	}

	long long int ed = cv::getTickCount();
	LOGI("end time: %lld", ed);
	double alltime = (double)(ed - st)/ cv::getTickFrequency();
	LOGI("use time: %f sec, image count: %d", alltime, len);
}

#else

static int TrackLanes(void* const userData, const cv::Mat &image,
		const std::string file)
{
	LaneTrackArg *arg = (LaneTrackArg*)userData;
	if( arg == 0 )
		return STEREO_ERR_PARAM;

	int res = (*arg)(image);

#if !ANDROID && !TEST_TIME
	int ch = cv::waitKey(line.debug ? 0: 5);
	if( ch == 'd' || ch == 'D' )
		line.debug = !line.debug;
	else if( ch == 27 )
		return STEREO_USE_EXIT;
#endif
	return res;
}

#endif


static int ParseCmdLine(CmdLine &line, int argc, char* argv[])
{
	static const char *shortopts = ":"
			"e:" // image file extension
			"l:" // left image file pattern
			"p:" // image files path and params save path
			"r:" // right image file pattern
			"i:" // source video file

			"c:" // car width
			"f:" // shift
			"u:" // bird view scale
			"k:" // skip frames
			"j:" // jump

			"D"  // debug
			"R"  // image is reactive
			"E"  // end ponit
			"T"  // Tachograph video model
			"S"  // save result

			"h"  // print help message
			"v";  // paint version

	struct option longopts[] = {
			{"ext", required_argument, NULL, 'e'},
			{"left", required_argument, NULL, 'l'},
			{"path", required_argument, NULL, 'p'},
			{"right", required_argument, NULL, 'r'},
			{"input", required_argument, NULL, 'i'},

			{"car_width", required_argument, NULL, 'c'},

			{"shift", required_argument, NULL, 'f'},
			{"unit", required_argument, NULL, 'u'},
			{"skip", required_argument, NULL, 'k'},
			{"jump", required_argument, NULL, 'j'},

			{"endpoint", no_argument, NULL, 'E'},

			{"debug", no_argument, NULL, 'D'},
			{"rectify", no_argument, NULL, 'R'},
			{"tri", no_argument, NULL, 'T'},
			{"save", no_argument, NULL, 'S'},

			{"help", no_argument, NULL, 'h'},
			{"version", no_argument, NULL, 'v'},
			{0, 0, 0, 0},
	};
	int res, idx;

	while( (res = getopt_long(argc, argv, shortopts, longopts, &idx)) != -1 )
	{
		switch(res)
		{
		case 'p':
			strcpy(line.path, optarg);
			break;
		case 'e':
			line.search.ext = optarg;
			break;
		case 'l':
			line.search.left_pattern = optarg;
			break;
		case 'r':
			line.search.rght_pattern = optarg;
			break;
		case 'i':
			strcpy(line.input, optarg);
			break;

		case 'c' :
			std::stringstream(optarg) >> line.car_width;
			break;
		case 'f' :
			std::stringstream(optarg) >> line.shift;
			break;
		case 'u' :
			std::stringstream(optarg) >> line.pixel_unit;
			break;
		case 'k':
			std::stringstream(optarg) >> line.skip;
			break;
		case 'j':
			std::stringstream(optarg) >> line.jump;
			break;

		case 'D':
			line.debug = 1;
			break;
		case 'E':
			line.ep = 1;
			break;
		case 'T':
			line.tri = 1;
			break;
		case 'S':
			line.save = 1;
			break;

		case 'h':
			print_help();
			return STEREO_USE_EXIT;
		case 'v':
			std::cout<<LaneTrack::GetVersion()<<std::endl;
			return STEREO_USE_EXIT;

		case ':': case '?': default:
			print_help();
			return STEREO_ERR_ARG;
		}
	}

	return STEREO_SUCCEED;
}

static void print_help()
{
    std::cout<<"Detect Road Lane. \n";

    std::cout<< "\nUsage: "
    		"\t lane [options] [-p path] \n"
    		"\t lane [--help][--version] \n";

    std::cout<< "\nOptions: \n"
    		"  -e, --ext        image file extension.\n"
    		"  -f, --shift      bird view shift y coordinate.\n"
    		"  -h, --help       print help message.\n"
    		"  -i, --input      input source image or video.\n"
    		"  -k, --skip       skip video frames.\n"
    		"  -l, --left       left image file pattern.\n"
    		"  -p, --path       image files path.\n"
    		"  -r, --right      right image file pattern.\n"
    		"  -s, --second     interval seconds.\n"
    		"  -S, --save       save result.\n"
    		"  -T, --tri        Tachograph video model.\n"
    		"  -u, --unit       bird view 1 pixel size, cm/pixel.\n"
    		"  -v, --version    print version and copyright info.\n"
    		<<std::endl;
}


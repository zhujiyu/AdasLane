#include "opencv2/core.hpp"

using namespace cv;

#undef R2Y
#undef G2Y
#undef B2Y

enum
{
    yuv_shift = 14,
    xyz_shift = 12,
    R2Y = 4899,
    G2Y = 9617,
    B2Y = 1868,
    BLOCK_SIZE = 256
};

const int ITUR_BT_601_CY  = 1220542;
const int ITUR_BT_601_CUB = 2116026;
const int ITUR_BT_601_CUG = -409993;
const int ITUR_BT_601_CVG = -852492;
const int ITUR_BT_601_CVR = 1673527;
const int ITUR_BT_601_SHIFT = 20;

#define MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION (320*240)


/// template<int bIdx, int uIdx, int yIdx> /// YUYV bIdx=0, uIdx=0, yIdx=0
struct YUV422toRGB888Invoker : ParallelLoopBody
{
    const uchar * src_data;
    uchar * dst_data;
    const size_t src_step, dst_step;
    const size_t width, height;

    YUV422toRGB888Invoker(uchar *_dst_data, const size_t _dst_step,
    		const uchar *_src_data, const size_t _src_step,
    		const size_t _width, const size_t _height):
    			src_data(_src_data), dst_data(_dst_data), src_step(_src_step),
    			dst_step(_dst_step), width(_width), height(_height)
	{}

    void operator()(const Range& range) const
    {
    	const int rangeBgn = range.start, rangeEnd = range.end;
    	const int sstep1 = src_step, sstep2 = src_step/2;

    	const int ustart = height * src_step, vstart = ustart + ustart/2;
        const uchar* ysrc = src_data + rangeBgn * sstep1;
        const uchar* usrc = src_data + ustart + rangeBgn * sstep2;
        const uchar* vsrc = src_data + vstart + rangeBgn * sstep2;

        for( int row = rangeBgn; row < rangeEnd; row++ )
        {
            uchar* dd = dst_data + dst_step * row;

            for( size_t col = 0; col < width; col += 2, dd += 6 )
            {
                int u = int(usrc[col>>1]) - 128;
                int v = int(vsrc[col>>1]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v+ ITUR_BT_601_CUG * u;
                int buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * u;

                int y00 = MAX(0, int(ysrc[col+0]) - 16) * ITUR_BT_601_CY;
                int y01 = MAX(0, int(ysrc[col+1]) - 16) * ITUR_BT_601_CY;

                dd[2] = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
                dd[1] = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
                dd[0] = saturate_cast<uchar>((y00 + buv) >> ITUR_BT_601_SHIFT);

                dd[5] = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
                dd[4] = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);
                dd[3] = saturate_cast<uchar>((y01 + buv) >> ITUR_BT_601_SHIFT);
            }

			ysrc += sstep1; usrc += sstep2; vsrc += sstep2;
        }
    }
};


struct cvtYellowGrayInvoker : cv::ParallelLoopBody
{
	const cv::Mat src;
	cv::Mat *dst;

	cvtYellowGrayInvoker(const cv::Mat src, cv::Mat *dst): src(src), dst(dst) {}

    uchar toYellowGray(int g, int r) const
    {
    	return saturate_cast<uchar>( (g + r + r + r) >> 2 );
    }
};

struct cvtBGR2YellowGrayInvoker : cvtYellowGrayInvoker
{
	cvtBGR2YellowGrayInvoker(const cv::Mat src, cv::Mat *dst):
		cvtYellowGrayInvoker(src, dst)
	{
        const int coeffs0[] = { R2Y, G2Y, B2Y };

        int b = 0, g = 0, r = (1 << (yuv_shift-1));
        int db = coeffs0[2], dg = coeffs0[1], dr = coeffs0[0];

        for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
        }
	}

    void operator()(const Range& range) const
    {
    	const int rowStep = src.rows/dst->rows;
    	const int sstep = src.cols/dst->cols*3;

        for( int row = range.start; row < range.end; row++ )
        {
            const uchar *sd = src.ptr<uchar>(row*rowStep);
            uchar *dd = dst->ptr<uchar>(row);

            for( int i = 0; i < dst->cols; i++, dd++, sd += sstep )
            	*dd = toYellowGray(sd[1], sd[2]);
        }
    }

    int tab[256*3];
};

struct cvtYUV422ToYellowGrayInvoker : cvtYellowGrayInvoker
{
	cvtYUV422ToYellowGrayInvoker(const cv::Mat src, cv::Mat *dst):
		cvtYellowGrayInvoker(src, dst)
	{}

    void operator()(const Range& range) const
    {
        const int ustart = src.rows * src.step.p[0], vstart = ustart/2 + ustart;
        const int rowStep = src.rows/ dst->rows, rangeBgn = range.start*rowStep;
        const int colStep = src.cols/ dst->cols;

        int sstep1 = src.step.p[0], sstep2 = src.step.p[0]/2;

        const uchar* ysrc = src.data + rangeBgn * sstep1;
        const uchar* usrc = src.data + ustart + rangeBgn* sstep2;
        const uchar* vsrc = src.data + vstart + rangeBgn* sstep2;

        sstep1 *= rowStep;
        sstep2 *= rowStep;

        if( colStep == 1 )
        {
			for( int row = range.start; row < range.end; row++ )
			{
				uchar* dd = dst->ptr<uchar>(row);

				for( int c = 0; c < dst->cols; c += 2, dd += 2 )
				{
					int y00 = MAX(0, int(ysrc[c+0]) - 16) * ITUR_BT_601_CY;
					int y01 = MAX(0, int(ysrc[c+1]) - 16) * ITUR_BT_601_CY;

					int u = int(usrc[c>>1]) - 128;
					int v = int(vsrc[c>>1]) - 128;

					int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
					int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v+ ITUR_BT_601_CUG * u;

					int r0 = (y00 + ruv) >> ITUR_BT_601_SHIFT;
					int g0 = (y00 + guv) >> ITUR_BT_601_SHIFT;

					int r1 = (y01 + ruv) >> ITUR_BT_601_SHIFT;
					int g1 = (y01 + guv) >> ITUR_BT_601_SHIFT;

					dd[0] = toYellowGray(g0, r0);
					dd[1] = toYellowGray(g1, r1);
				}

				ysrc += sstep1; vsrc += sstep2; usrc += sstep2;
			}
        }
        else
        {
			for( int row = range.start; row < range.end; row++ )
			{
				uchar* dd = dst->ptr<uchar>(row);

				for( int c = 0; c < dst->cols; c++, dd++ )
				{
					int sc = c*colStep;

					int u = int(usrc[sc>>1]) - 128;
					int v = int(vsrc[sc>>1]) - 128;
					int y00 = MAX(0, int(ysrc[sc+0]) - 16) * ITUR_BT_601_CY;

					int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
					int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v+ ITUR_BT_601_CUG * u;

					int r0 = (y00 + ruv) >> ITUR_BT_601_SHIFT;
					int g0 = (y00 + guv) >> ITUR_BT_601_SHIFT;

					*dd = toYellowGray(g0, r0);
				}

				ysrc += sstep1; vsrc += sstep2; usrc += sstep2;
			}
        }
    }
};

void SubYUV422Image(const cv::Mat src, cv::Mat dst, const cv::Rect rect)
{
	const int sw = src.cols, sh = src.rows/2, rw = rect.width, rh = rect.height;
	const int _ysrccnt = src.step.p[0] * sh, _usrccnt = _ysrccnt/2;
	const int _ydstcnt = rect.width * rect.height, _udstcnt = _ydstcnt/2;

	const cv::Mat ysrc(sh, sw  , CV_8UC1, src.data);
	const cv::Mat usrc(sh, sw/2, CV_8UC1, src.data + _ysrccnt);
	const cv::Mat vsrc(sh, sw/2, CV_8UC1, src.data + _ysrccnt + _usrccnt);

	cv::Rect uvrect(rect.x/2, rect.y, rect.width/2, rect.height);
	dst.create(rh *2, rw, CV_8UC1);

	ysrc(  rect).copyTo(cv::Mat(rect.height, rect.width  , CV_8UC1, dst.data));
	usrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, dst.data + _ydstcnt));
	vsrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, dst.data + _ydstcnt + _udstcnt));
}

void cvtYUV422toYellowGray(const cv::Mat src, cv::Mat &gray, const cv::Rect rect)
{
	if( gray.empty() )
		return;
	cv::Mat subsrc(rect.size(), CV_8UC1);
	SubYUV422Image(src, subsrc, rect);

	cvtYUV422ToYellowGrayInvoker cvt(subsrc, &gray);

    if( rect.width * rect.height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, gray.rows), cvt);
    else
    	cvt(Range(0, gray.rows));
}

void cvtBGR2YellowGray(const cv::Mat src, cv::Mat &gray, const cv::Rect rect)
{
	if( gray.empty() )
		return;
	cvtBGR2YellowGrayInvoker cvt(src(rect), &gray);

    if( rect.width * rect.height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, gray.rows), cvt);
    else
    	cvt(Range(0, gray.rows));
}

void cvtYUV422toBGR(const cv::Mat src, cv::Mat &bgr, const cv::Rect rect)
{
	cv::Mat _src = src(rect);
	YUV422toRGB888Invoker cvt(bgr.data, bgr.step.p[0],
			_src.data, _src.step.p[0], _src.cols, _src.rows);

    if( rect.width * rect.height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, bgr.rows), cvt);
    else
    	cvt(Range(0, bgr.rows));
}


struct toYellowGrayByRInvoker : ParallelLoopBody
{
    const uchar * src_data;
    uchar * dst_data;
    const size_t src_step, dst_step;
    const size_t width, height, rowStep, colStep;

    toYellowGrayByRInvoker(uchar *_dst_data, const size_t _dst_step,
    		const uchar *_src_data, const size_t _src_step,
    		const size_t _width, const size_t _height,
			const size_t _rowStep, const size_t _colStep):
    			src_data(_src_data), dst_data(_dst_data), src_step(_src_step),
    			dst_step(_dst_step), width(_width), height(_height),
				rowStep(_rowStep), colStep(_colStep)
	{}

    uchar toYellowScale(int g, int r) const
    {
    	return saturate_cast<uchar>( (g + r + r + r) >> 2 );
    }
};

inline int Align(int x, int n)
{
	return ((x - 1)/n + 1) * n;
}

struct BGR2YellowGrayInvoker : toYellowGrayByRInvoker
{
	BGR2YellowGrayInvoker(uchar *_dst_data, const size_t _dst_step,
			const uchar *_src_data, const size_t _src_step,
			const size_t _width, const size_t _height,
			const size_t _rowStep = 1, const size_t _colStep = 1):
				toYellowGrayByRInvoker(_dst_data, _dst_step,  _src_data,
						_src_step, _width, _height, _rowStep, _colStep)
    {
        const int coeffs0[] = { R2Y, G2Y, B2Y };

        int b = 0, g = 0, r = (1 << (yuv_shift-1));
        int db = coeffs0[2], dg = coeffs0[1], dr = coeffs0[0];

        for( int i = 0; i < 256; i++, b += db, g += dg, r += dr )
        {
            tab[i] = b;
            tab[i+256] = g;
            tab[i+512] = r;
        }
    }

    void operator()(const Range& range) const
    {
    	const int rangeBgn = Align(range.start, rowStep);
    	const int rangeEnd = Align(range.end, rowStep);

//        const int rangeBegin = (range.start / rowStep + (range.start % rowStep ? 1 : 0)) * rowStep;
//        const int rangeEnd = (range.end / rowStep + (range.end % rowStep ? 1 : 0)) * rowStep;

        for (int row = rangeBgn; row < rangeEnd; row += rowStep )
        {
            const uchar *src = src_data + src_step* row;
            uchar* dst = dst_data + dst_step * row;

            for( size_t i = 0; i < width; i++, dst++, src += 3 )
            	*dst = toYellowScale(src[1], src[2]);
        }
    }

    int tab[256*3];
};


struct YUV422toYellowGrayInvoker : toYellowGrayByRInvoker
{
    YUV422toYellowGrayInvoker(uchar *_dst_data, const size_t _dst_step,
    		const uchar *_src_data, const size_t _src_step, const size_t _width,
			const size_t _height, const size_t _rowStep = 1, const size_t _colStep = 1):
				toYellowGrayByRInvoker(_dst_data, _dst_step, _src_data, _src_step,
						_width, _height, _rowStep, _colStep)
    {}

    void operator()(const Range& range) const
    {
    	const int rangeBgn = Align(range.start, rowStep);
    	const int rangeEnd = Align(range.end, rowStep);

//        const int rangeBegin = (range.start/rowStep + (range.start%rowStep?1: 0))*rowStep;
//        const int rangeEnd = (range.end/rowStep + (range.end%rowStep?1: 0))*rowStep;

        const int ustart = height * src_step, vstart = ustart + height * src_step/2;
        const int sstep1 = src_step, sstep2 = src_step/2;

        const uchar* ysrc = src_data + rangeBgn * sstep1;
        const uchar* usrc = src_data + ustart + rangeBgn* sstep2;
        const uchar* vsrc = src_data + vstart + rangeBgn* sstep2;

        for (int row = rangeBgn; row < rangeEnd; row += rowStep )
        {
            uchar* rowData = dst_data + dst_step * row;

            for( size_t i = 0; i < width; i += 2, rowData += 2 )
            {
                int y00 = MAX(0, int(ysrc[i+0]) - 16) * ITUR_BT_601_CY;
                int y01 = MAX(0, int(ysrc[i+1]) - 16) * ITUR_BT_601_CY;

                int u = int(usrc[i>>1]) - 128;
                int v = int(vsrc[i>>1]) - 128;

                int ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * v;
                int guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * v+ ITUR_BT_601_CUG * u;

                int r0 = (y00 + ruv) >> ITUR_BT_601_SHIFT;
                int g0 = (y00 + guv) >> ITUR_BT_601_SHIFT;

                int r1 = (y01 + ruv) >> ITUR_BT_601_SHIFT;
                int g1 = (y01 + guv) >> ITUR_BT_601_SHIFT;

                rowData[0] = toYellowScale(g0, r0);
                rowData[1] = toYellowScale(g1, r1);
            }

			ysrc += sstep1; vsrc += sstep2; usrc += sstep2;
        }
    }
};


inline void cvtYUV422toYellowGray(uchar *dst_data, const size_t dst_step,
		const uchar *src_data, const size_t src_step,
		const size_t width, const size_t height, const size_t rowStep)
{
	YUV422toYellowGrayInvoker converter(dst_data, dst_step, src_data,
			src_step, width, height, rowStep);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}

cv::Mat cvtYUV422toYellowGray(const uchar* src_data, const cv::Size size,
		const size_t rowStep)
{
	cv::Mat gray(size, CV_8UC1, cv::Scalar::all(0));
	cvtYUV422toYellowGray(gray.data, size.width, src_data,
			size.width, size.width, size.height, rowStep);
	return gray;
}

cv::Mat cvtYUV422toYellowGray(uchar *src_data, const cv::Size size,
		const cv::Rect rect, const size_t rowStep)
{
	int _ysrccnt = size.width*size.height, _usrccnt = _ysrccnt/2;
	int _ydstcnt = rect.width*rect.height, _udstcnt = _ydstcnt/2;
	cv::Rect uvrect(rect.x/2, rect.y, rect.width/2, rect.height);
	uchar _src_data[rect.width*rect.height*2];

	const cv::Mat ysrc(size.height, size.width  , CV_8UC1, src_data);
	const cv::Mat usrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt);
	const cv::Mat vsrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt + _usrccnt);

	ysrc(  rect).copyTo(cv::Mat(rect.height, rect.width  , CV_8UC1, _src_data));
	usrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt));
	vsrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt + _udstcnt));

	return cvtYUV422toYellowGray(_src_data, rect.size(), rowStep);
}


inline void cvtBGR2YellowGray(uchar *dst_data, const size_t dst_dataStep,
		const uchar *src_data, const size_t src_dataStep,
		const size_t width, const size_t height, const size_t rowStep)
{
	BGR2YellowGrayInvoker converter(dst_data, dst_dataStep,
			src_data, src_dataStep, width, height, rowStep);
//    converter(Range(0, height));
//	sjsParallel(Range(0, height), converter);
	// parallel_for_(Range(0, height), converter);
    if( width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION )
   	    parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}

cv::Mat cvtBGR2YeloowGray(const cv::Mat src)
{
	cv::Mat gray(src.size(), CV_8UC1, cv::Scalar::all(0));
	cvtBGR2YellowGray(gray.data, gray.step, (uchar*)src.ptr<cv::Vec3b>(0),
			src.step, src.cols, src.rows, 1);
	return gray;
}

cv::Mat cvtBGR2YeloowGray(const cv::Mat src, const cv::Rect rect, const int rowStep)
{
	cv::Mat gray(rect.size(), CV_8UC1, cv::Scalar::all(0));
	cvtBGR2YellowGray(gray.data, gray.step, (uchar*)src.ptr<cv::Vec3b>(0)
			+ src.step * rect.y + rect.x*3, src.step, rect.width, rect.height, rowStep);
	return gray;
}


inline void cvtYUV422toRGB(uchar *dst_data, size_t dst_step, const uchar *src_data, const size_t src_step,
                           const int width, const int height)
{
    YUV422toRGB888Invoker converter(dst_data, dst_step, src_data, src_step, width, height);
    if (width * height >= MIN_SIZE_FOR_PARALLEL_YUV422_CONVERSION)
        parallel_for_(Range(0, height), converter);
    else
        converter(Range(0, height));
}


void cvtYUV422toBGR(cv::Mat &bgr, const uchar* src_data, const cv::Size size, size_t stride)
{
	// cv::Mat gray(size, CV_8UC3, cv::Scalar::all(0));
	if( stride == 0 )
		stride = size.width;
	bgr.create(size, CV_8UC3);
	cvtYUV422toRGB(bgr.data, bgr.step, src_data, stride, size.width, size.height);
}

cv::Mat cvtYUV422toBGR(const uchar* src_data, const cv::Size size)
{
	cv::Mat gray(size, CV_8UC3, cv::Scalar::all(0));
	cvtYUV422toRGB(gray.data, gray.step, src_data, size.width, size.width, size.height);
	return gray;
}

cv::Mat cvtYUV422toBGR(uchar *src_data, const cv::Size size, const cv::Rect rect)
{
	int _ysrccnt = size.width*size.height, _usrccnt = _ysrccnt/2;
	int _ydstcnt = rect.width*rect.height, _udstcnt = _ydstcnt/2;
	cv::Rect uvrect(rect.x/2, rect.y, rect.width/2, rect.height);
	uchar _src_data[rect.width*rect.height*2];

	const cv::Mat ysrc(size.height, size.width  , CV_8UC1, src_data);
	const cv::Mat usrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt);
	const cv::Mat vsrc(size.height, size.width/2, CV_8UC1, src_data + _ysrccnt + _usrccnt);

	ysrc(  rect).copyTo(cv::Mat(rect.height, rect.width  , CV_8UC1, _src_data));
	usrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt));
	vsrc(uvrect).copyTo(cv::Mat(rect.height, rect.width/2, CV_8UC1, _src_data + _ydstcnt + _udstcnt));

	return cvtYUV422toBGR(_src_data, rect.size());
}


//    uchar toYellowScale(uchar gray, uchar r, uchar g, uchar b) const
//    {
//    	return saturate_cast<uchar>(r > gray ? r*2 - gray: r);
//    }
//
//    uchar toYellowScale(uchar g, uchar r) const
//    {
//    	return saturate_cast<uchar>( (g + r + r + r) >> 2 );
//    }

//                int r0 = saturate_cast<uchar>((y00 + ruv) >> ITUR_BT_601_SHIFT);
//                int g0 = saturate_cast<uchar>((y00 + guv) >> ITUR_BT_601_SHIFT);
//
//                int r1 = saturate_cast<uchar>((y01 + ruv) >> ITUR_BT_601_SHIFT);
//                int g1 = saturate_cast<uchar>((y01 + guv) >> ITUR_BT_601_SHIFT);

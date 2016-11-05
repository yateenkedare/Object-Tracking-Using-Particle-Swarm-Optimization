#include "HSVHistogram.h"
#include <iostream>
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;
void calc_hsvHistogram(Mat& inpFrame, Mat& outHist) {
	
	static const int h_bins = 8; 
	//static const int s_bins = 8;
	//static const int v_bins = 10;
	static const int histSize[] = { h_bins};

	static const float h_ranges[] = { 0, 256 };
	//static const float s_ranges[] = { 0, 256 };
	//static const float v_ranges[] = { 0, 256 };
	static const float* ranges[] = { h_ranges};

	static const int channels[] = { 0};
	Mat srcs[] = {inpFrame};

	calcHist(srcs,sizeof(srcs),channels,Mat(),outHist,1,histSize,ranges,true,false);
	normalize(outHist,outHist);	//to get the probability density
}
/*

TODO PSO-
	1. Make delta dynamic according to the input size selected
	2. Tune the position affecting factors c1,c2,w
	3. find better conditions for changing the width and height of the particles
	
*/
#include<iostream>
#include <opencv2\opencv.hpp>
#include"state.h"
#include"selector.h"
#include "ParticleSwarmOptimizer.h"
#include"HSVHistogram.h"
#include"Particles.h"
using namespace cv;
using namespace std;

const char* WINDOW  = "PSO";

const Scalar RED = Scalar(0, 0, 255);
const Scalar BLUE = Scalar(255, 0, 0);
const Scalar GREEN = Scalar(0, 255, 0);
const Scalar YELLOW = Scalar(0, 255, 255);
const Scalar WHITE = Scalar(255, 255, 255);

inline void update_target_histogram(Mat& image, Mat& HSVimage, Rect& selection, Mat& histogram , Mat& target)
{
	//Calculate target Histogram and update target_histogram
	Mat HSVroi(HSVimage, selection);
	calc_hsvHistogram(HSVroi, histogram);
	cout<<endl<<image.cols<<image.rows<<endl;
	/*
	Mat hist;
	calc_hsvHistogram(HSVimage,hist);
	float bc = compareHist(hist,histogram, CV_COMP_BHATTACHARYYA);
	cout<<"BC : "<<bc<<endl;*/
	
	//for displaying RGB target image
	if(!target.empty())	target.release();
	Mat roi(image, selection);
	roi.copyTo(target);
	cvDestroyWindow("Target object");
	imshow("Target object", target);
}

struct StateData {
	StateData(int num_particles):
		image(),
		HSVimage(),
		target(),
		target_histogram(),
		selector(WINDOW), 
		selection(),
		paused(false),
		draw_particles(false),
		PSO(num_particles),
		delta(0)
	{};

	Mat image;
	Mat HSVimage;
	Mat target;
	Mat target_histogram;
	Selector selector;
	Rect selection;
	bool paused;
	bool draw_particles;
	ParticleSwarmOptimizer PSO;
	int delta;
};

State_ state_start(StateData& d) {

	if( d.selector.selecting() ) {  
		cout << "state_selecting" << endl;
		return state_selecting;
	}
	else {
		return state_start;
	}
}



State_ state_selecting(StateData& d) {
	if( d.selector.valid() ) {
		cout << "state_initializing: (" << d.selection.x << ", " << d.selection.y << ", " << d.selection.width << ", " << d.selection.height  << ")" << endl;
		d.selection = d.selector.selection();
		cout << "selection: (" << d.selection.x << ", " << d.selection.y << ", " << d.selection.width << ", " << d.selection.height  << ")" << endl;
		return state_initializing(d); // Call with current frame
	}
	else {
		Mat roi(d.image, d.selector.selection());
		bitwise_not(roi, roi);
		return state_selecting;
	}
}
/*
State_ state_surf(StateData& d) {
	if (detectUsingSurf(d.image, d.selection) )
			return	state_initializing;
	   else {
		d.pause = false;
			 return state_start;
	}

	
}*/

State_ state_initializing(StateData& d) {
	if( d.selector.selecting() ) {  
		cout << "state_selecting" << endl;
		return state_selecting;
	}

	// Generate initial target histogram
	update_target_histogram(d.image, d.HSVimage, d.selection, d.target_histogram, d.target);
	
	// Initialize Particles with center of selection
	d.PSO.init(d.selection,d.HSVimage,d.target_histogram);

	// Start video running if paused
	d.paused = false;

	cout << "state_tracking" << endl;
	return state_tracking(d); // Call with current frame
}

State_ state_tracking(StateData& d) {
	if( d.selector.selecting() ) {  
		cout << "state_selecting" << endl;
		return state_selecting;
	}
	//Update particles
	d.PSO.UpdateVelocityAndPosition(d.image, d.HSVimage,d.target_histogram);

	// Draw particles
	if( d.draw_particles )
		d.PSO.drawParticles(d.image, BLUE);
	
	return state_tracking;
}


struct Options {
	Options()
		:num_particles(200),
		use_lbp(false),
		infile(),
		outfile()
	{}

	int num_particles;
	bool use_lbp;
	string infile;
	string outfile;
};

void main() {
	cv::setBreakOnError(true);				//For debugging purpose
	Options o;
	o.outfile = "out.avi";
	//o.infile = "02291vFF.avi";
	VideoCapture cap;
	VideoWriter writer;
	
	bool use_camera = false;

	// Use filename if given, else use default camera
	if(!o.infile.empty()) {
		cap.open(o.infile);
		use_camera = false;
	}
	else {
		cap.open(0);
		use_camera = true;
	}

	if( !cap.isOpened() ) {
		cerr << "Failed to open capture device" << endl;
		exit(EXIT_FAILURE);
	}

	if( !o.outfile.empty() ) {
		int fps = cap.get(CV_CAP_PROP_FPS);
		int width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
		int height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
		writer.open(o.outfile,CV_FOURCC('j', 'p', 'e', 'g'), 25, Size(width, height),true);
		if( !writer.isOpened() ) {
			cerr << "Could not open '" << o.outfile << "'" << endl;
			exit(EXIT_FAILURE);
			use_camera = false;
		}

	}
	
	// Open window and start capture
	namedWindow(WINDOW);
	
	StateData d(o.num_particles);

	State state = state_start;

	Mat frame;

	
	for(;;) {
		if( !d.paused) {
			cap >> frame;
			if(frame.empty()) {
				cerr << "Error reading frame" << endl;
				break;
			}
			
		}

		if( use_camera ) {
			//resize(frame,frame, Size(200, 200), INTER_NEAREST);
			flip(frame, d.image, 1);
		}
		else {
			frame.copyTo(d.image);
		}

		char c = (char)waitKey(1);
		if( c == 27 )
			break;
		switch(c)
		{
		case 'p':
			d.paused = !d.paused;
			break;

		case 'c':
			cout << "Tracking cancelled." << endl;
			state = state_start;
			break;

		case 'd':
			d.draw_particles = !d.draw_particles;
			cout << "Draw particles: " << d.draw_particles << endl;
			break;

		case 's':
			cout << "Object Detection" << endl;
			d.paused = true;
		//	state = state_surf;
			break;
		}

		//Converting image to HSV 
		cvtColor(d.image,d.HSVimage,CV_BGR2HSV);

		// Process frame in current state
		state = state(d);
		
		imshow(WINDOW, d.image);
		if( writer.isOpened() && !d.paused )
		{
			writer << d.image;
		}
	}

}
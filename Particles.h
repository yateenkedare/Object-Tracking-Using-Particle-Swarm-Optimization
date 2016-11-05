#ifndef PARTICLES_H
#define PARTICLES_H
#include<vector>
#include<time.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
class Particles
{
public:
	Particles(int);			
	~Particles();
	void initParticles(Rect&,int,int);		//num_particles , position of the selected target, rows, cols
	
protected:
	uint64 initValue;
	RNG rng;
	int num_particles;
	vector<float> l_best;	//local best of the particle
	vector<int> l_best_x;
	vector<int> l_best_y;
	vector<int> l_best_h;
	vector<int> l_best_w;
	vector<Rect> position;		//x,y,length, width of the particle
	vector<float> fitness;	//current fitness of the particle
	vector<float> velocityX;	//current velocity vector of particle
	vector<float> velocityY;	//current velocity vector of particle
	vector<Mat> particleHistogram;	//HSV histogram of current particle
	static const int delta = 25;   //delta for redistribution if delta = 50 particles will be redistributed over a scale of 100 pixels

};


#endif
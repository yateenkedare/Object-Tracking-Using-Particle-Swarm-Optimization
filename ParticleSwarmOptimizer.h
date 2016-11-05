#ifndef PARTICLESWARMOPTIMIZER_h
#define PARTICLESWARMOPTIMIZER_h
#include"Particles.h"
#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class ParticleSwarmOptimizer : private Particles
{
public :
	ParticleSwarmOptimizer(int n);
	virtual ~ParticleSwarmOptimizer();
	void init(Rect&, Mat&,Mat& targetHist);					//selection, image
	void UpdateVelocityAndPosition(Mat& image, Mat& HSVimage,Mat& targetHist);
	void drawParticles(Mat&,const Scalar& color);
	int round(float number);
	void print_to_file();

private :
	int g_best_pos;											//global best of all the particles
	int g_best_x;
	int g_best_y;
	int g_best_w;
	int g_best_h;
	float calc_fitness(Mat& inp1, Mat& inp2);		//calculate current fitness of the particle
	
};

#endif
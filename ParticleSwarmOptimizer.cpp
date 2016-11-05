#include "ParticleSwarmOptimizer.h"
#include "HSVHistogram.h"
#include "state.h"
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
typedef unsigned int uint;
//int errnum = 0;
ParticleSwarmOptimizer::ParticleSwarmOptimizer(int n)
	:Particles(n)
{
}

ParticleSwarmOptimizer::~ParticleSwarmOptimizer() {}

void ParticleSwarmOptimizer::init(Rect& pos, Mat& HSVimage,Mat& targetHist)	{
	initParticles(pos, HSVimage.rows, HSVimage.cols);			//creating particles and assigning random velocity to each particle
	
	for(uint i = 0; i<num_particles;i++) {
		particleHistogram[i].release();
		calc_hsvHistogram(Mat(HSVimage,position[i]),particleHistogram[i]);		//calculating histogram of each particle
		fitness[i] = calc_fitness(particleHistogram[i], targetHist);	//finding the fitness of each particle 
		l_best[i] = fitness[i];
		l_best_x[i] = position[i].x;
		l_best_y[i] = position[i].y;
		l_best_w[i] = position[i].width;
		l_best_h[i] = position[i].height;
	}
	g_best_pos = distance(l_best.begin(), min_element(l_best.begin(), l_best.end()));		//store the position of the best fitness in g_best_pos
	g_best_x = l_best_x[g_best_pos];
	g_best_y = l_best_y[g_best_pos];
	g_best_w = position[g_best_pos].width;
	g_best_h = position[g_best_pos].height;
	//print_to_file();
}

void ParticleSwarmOptimizer::drawParticles(Mat& image, const Scalar& color) {
	for(uint i = 0; i<num_particles;i++) {
		rectangle(image,position[i],color,1);
		//circle(image,Point(position[i].x + (position[i].width)/2, position[i].y + (position[i].height)/2),1,color);
	}
}

float ParticleSwarmOptimizer::calc_fitness(Mat& inp1, Mat& inp2) {
	return compareHist(inp1, inp2, CV_COMP_BHATTACHARYYA) * 100;
}

void ParticleSwarmOptimizer::UpdateVelocityAndPosition(Mat& image, Mat& HSVimage,Mat& targetHist) {
	
	//Rect bound(0,0,HSVimage.cols,HSVimage.rows);
	
	for(uint i = 0; i<num_particles;i++) {
		particleHistogram[i].release();
		calc_hsvHistogram(Mat(HSVimage,position[i]),particleHistogram[i]);		//calculating histogram of each particle
		fitness[i] = calc_fitness(particleHistogram[i], targetHist);	//finding the fitness of each particle 
		if (fitness[i]<l_best[i]) { 
			l_best[i] = fitness[i];					
			l_best_x[i] = position[i].x;
			l_best_y[i] = position[i].y;
			l_best_w[i] = position[i].width;
			l_best_h[i] = position[i].height;
		}
	
	}
	g_best_pos = distance(fitness.begin(), min_element(fitness.begin(), fitness.end()));		//store the position of the best fitness in g_best_pos
	g_best_x = position[g_best_pos].x;
	g_best_y = position[g_best_pos].y;
	g_best_w = position[g_best_pos].width;
	g_best_h = position[g_best_pos].height;
	rectangle(image,position[g_best_pos],Scalar(0,255,0),4);
	circle(image,Point(position[g_best_pos].x + (position[g_best_pos].width)/2, position[g_best_pos].y + (position[g_best_pos].height)/2),5,Scalar(0,0,255));

	const float c1 = 0.2,c2 =1, w = 1, c3 =0.45,c4 = 1.1;
	float newVelocity,newSize,r1,r2,r3,r4;
	for(uint i = 0; i < num_particles; i++) {
		
		r1 = rng.uniform(0.f,1.f);
		r2 = rng.uniform(0.f,1.f);
		
		newSize = position[i].width + c3*r1*(l_best_w[i] - position[i].width) + c4*r2*(g_best_w - position[i].width);
		if(newSize < 3)	position[i].width = 3;
		else position[i].width = newSize;
		
		newSize = position[i].height+ c3*r1*(l_best_h[i] - position[i].height) + c4*r2*(g_best_h - position[i].height);
		if(newSize < 3)	position[i].height= 3;
		else position[i].height= newSize;
		
		

		/*if(position[i].width > g_best_w)  position[i].width--;
		else if(position[i].width < g_best_w)position[i].width++;
		if(position[i].height > g_best_h)  position[i].height--;
		else if(position[i].height < g_best_h) position[i].height++;
		*/
		//updating velocity in x direction
		r1 = rng.uniform(0.f,1.f);
		r2 = rng.uniform(0.f,1.f);
		newVelocity =	w * velocityX[i] 
						+ c1 * r1 * (l_best_x[i] - position[i].x) 
						+ c2 * r2 * (g_best_x - position[i].x);
		newVelocity = round(newVelocity);
		if(newVelocity>50)	newVelocity = 50;
		if(newVelocity < -50) newVelocity = -50;
	    velocityX[i] = newVelocity;
		position[i].x += velocityX[i];
		
		if(position[i].x < 0)
			position[i].x = 0;
		else if(position[i].x > (HSVimage.cols - position[i].width))
			position[i].x = HSVimage.cols - position[i].width - 1;

		//updating velocity in y direction
		r3 = rng.uniform(0.f,1.f);
		r4 = rng.uniform(0.f,1.f);
		newVelocity =	w * velocityY[i] 
						+ c1 * r3 * (l_best_y[i] - position[i].y) 
						+ c2 * r4 * (g_best_y - position[i].y);
		newVelocity = round(newVelocity);
		if(newVelocity>50)	newVelocity = 50;
		if(newVelocity < -50) newVelocity = -50;
		velocityY[i] = newVelocity;
		position[i].y += velocityY[i];
		
		if(position[i].y < 0)
			position[i].y = 0;
		else if(position[i].y > (HSVimage.rows - position[i].height))
			position[i].y = HSVimage.rows - position[i].height - 1;
		//position[i] = position[i] & bound;
	}
	//print_to_file();
}

int ParticleSwarmOptimizer::round(float number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}

void ParticleSwarmOptimizer::print_to_file()	{
	ofstream myfile;
	myfile.open ("example.csv",fstream::app);
	
	for(int i = 0; i < num_particles; i++) {
		myfile<<i<<','<<position[i].x<<','<<position[i].y<<','<<position[i].width<<','<<position[i].height<<','<<velocityX[i]<<','<<velocityY[i]<<','<<fitness[i]<<','<<g_best_pos<<','<<l_best_x[i]<<','<<l_best_y[i]<<','<<g_best_x<<','<<g_best_y<<endl;
	}
	myfile.close();
}
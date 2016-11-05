#include "Particles.h"
#include <opencv2\opencv.hpp>
using namespace cv;
using namespace std;

Particles::Particles(int n) : 
	particleHistogram(n),
	position(n),
	l_best(n),
	fitness(n),
	velocityX(n),
	velocityY(n),
	l_best_x(n),
	l_best_y(n),
	l_best_h(n),
	l_best_w(n),
	initValue(time(0)),
	rng(initValue)
{
	num_particles = n;
	
}

Particles::~Particles() {
	
}

void Particles::initParticles( Rect& pos, int rows, int cols) {
	
	//TODO - create new particles
	//Initialise those particles at the initial target selectied position and assign them random velocities 

	int x,y,w,h;
	//Creating particles 
	for(unsigned int i = 0; i < num_particles; i++)	{

		//assigning initial position w.r.t the selected area
		w = rng.uniform(((pos.width - delta) < 5) ? 5:pos.width - delta, pos.width + delta );
		h = rng.uniform(((pos.height - delta) < 5) ?5:pos.height - delta , pos.height + delta);
		x = rng.uniform((pos.x - delta < 0) ? 0:pos.x - delta , (pos.x + delta > cols-w) ? cols-w: pos.x + delta);
		y = rng.uniform((pos.y - delta < 0) ? 0:pos.y - delta , (pos.y + delta > rows-h) ? rows-h: pos.y + delta);
		position[i] = Rect(x,y,w,h);

		//assigning random velocity to the particles
		velocityX[i] = rng.uniform(-5,5);
		velocityY[i] = rng.uniform(-5,5);
	}	

}
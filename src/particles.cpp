#include <laser_based_localization_pf/particles.h>


Particle::Particle(){
}


double Particle::getX(){
   return pose_.position.x;
}

double Particle::getY(){
    return pose_.position.y;
}

double Particle::getTheta(){
    return tf::getYaw(pose_.orientation);
}

void Particle::updatePose(double x, double y, double theta){
    pose_.position.x = x;
    pose_.position.y = y;

    tf::Quaternion q = tf::createQuaternionFromRPY(0 , 0, theta);
    pose_.orientation.w = q.getW();
    pose_.orientation.x = q.getX();
    pose_.orientation.y = q.getY();
    pose_.orientation.z = q.getZ();
}

void Particle::updateWeight(double weight){
    weight_ = weight;
}

#ifndef LASER_BASED_LOCALIZATION_PF_H
#define LASER_BASED_LOCALIZATION_PF_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>

#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <boost/thread.hpp>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/GetMap.h>
#include <nav_msgs/OccupancyGrid.h>

#include <Eigen/Dense>
#include "particles.h"

using namespace std;

class LaserBasedLocalizationPf
{
public:
    LaserBasedLocalizationPf(ros::NodeHandle n);

    void updateOdometry(nav_msgs::Odometry odometry);
    void updateLaser(sensor_msgs::LaserScan laser);
    void resetLocalization(double x, double y, double theta);


    void visualizeSeenLaser(sensor_msgs::LaserScan laser);
    void updateLocalization(Eigen::MatrixXd x, std::vector<Particle> &particles);

    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg);
    void odometryCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void initialposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg);
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg);


private:

    boost::mutex* data_mutex_;
    tf::TransformListener* tf_listener_;
    nav_msgs::Odometry last_odometry;
    nav_msgs::OccupancyGrid occ_grid_;
    sensor_msgs::LaserScan laser_info_;

    ros::NodeHandle nh_;

    Eigen::MatrixXd x;

    //samples for particel filter
    int num_particles_;
    std::vector<Particle> particles_;
    int max_x_position_;
    int max_y_position_;

    vector<int> distanceGrid_;
    nav_msgs::OccupancyGrid likelihood_; 


    //publsiher for particles
    ros::Publisher particles_pub_;
    ros::Publisher pose_with_cov_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher vis_pub_ ;
    ros::Publisher laser_pub_;
    ros::Publisher real_laser_pub_;
    tf::TransformBroadcaster pose_tf_broadcaster_;
    void publishParticles(std::vector<Particle>& particles);
    std::vector<geometry_msgs::Pose> getParticlePositions(std::vector<Particle>& particles);
    void publishPose(Eigen::MatrixXd& x, std::vector<Particle>& particles);
    void generateUncertaintyMarker(visualization_msgs::Marker& marker, Eigen::Matrix2f uncertainly_mat, Eigen::Vector2f position);

    //functions for Particle filter
    void initParticles();
    bool isValid(Particle p, nav_msgs::OccupancyGrid map);
    std::vector<int> computeDistanceGrid(nav_msgs::OccupancyGrid map);
    void normalizeParticleWeights();
    void resamplingParticles();
    double probNormalDistribution(double a, double variance);
    double sampleNormalDistribution(double variance);

    /*
     *  Ray tracing at the given position (x,y,theta)
     *  speedup - parameter defining how many angles are simulated
     *  i.e. speedup = 2 - every second angle is simulated
     */
    sensor_msgs::LaserScan::Ptr simulateLaser(double x, double y, double theta, double speedup = 1);


};

template<typename to, typename from>to arg_cast(from const &x) {
  std::stringstream os;
  to ret;

  os << x;
  os >> ret;

  return ret;
}


#endif // LASER_BASED_LOCALIZATION_H

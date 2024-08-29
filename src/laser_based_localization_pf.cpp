#include <laser_based_localization_pf/laser_based_localization_pf.h>
#include <laser_based_localization_pf/particles.h>
#include <occupancy_grid_utils/ray_tracer.h>

#include <geometry_msgs/PoseArray.h>
#include <ros/ros.h>
#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <fstream>
#include <random>

using namespace std;
using namespace ros;

const int INF = numeric_limits<int>::max();
bool is_moving = false;
vector<double> likelihood;

LaserBasedLocalizationPf::LaserBasedLocalizationPf(ros::NodeHandle n)
{
    data_mutex_ = new boost::mutex();
    tf_listener_ = new tf::TransformListener(n);

    nh_ = n;

    //Publisher for particles
    particles_pub_ = nh_.advertise<geometry_msgs::PoseArray>("particles", 100);
    pose_with_cov_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped>("robot_pose_with_cov",100);
    pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("robot_pose_",100);
    vis_pub_ = nh_.advertise<visualization_msgs::Marker>( "uncertainty_marker", 100 );
    laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>("simulated_laser",100);
    real_laser_pub_ = nh_.advertise<sensor_msgs::LaserScan>("real_laser",100);
    x = Eigen::MatrixXd::Zero(3,1);

    //initialize Particles here
    num_particles_ = 100;
    initParticles();
}

void LaserBasedLocalizationPf::initParticles()
{
    ros::ServiceClient map_client = nh_.serviceClient<nav_msgs::GetMap>("static_map");
    ros::service::waitForService("static_map");

    //get map from map server
    nav_msgs::GetMap srv;
    if(!map_client.call(srv))
    {
        ROS_ERROR("Not able to get map from map server!");
        ros::shutdown();
    }
    nav_msgs::OccupancyGrid map = srv.response.map;

    //get max x and y values - use them to distribute your particles over the whole map
    max_y_position_ = static_cast<int>(map.info.height * map.info.resolution);
    max_x_position_ = static_cast<int>(map.info.width  * map.info.resolution);

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis_x(0.0, max_x_position_);
    uniform_real_distribution<> dis_y(0.0, max_y_position_);


    // 1.1 can Instanciating particles
    while(particles_.size() < num_particles_){
        // Coords of the particle
        double x = dis_x(gen) + map.info.origin.position.x;   // To allineate the offset
        double y = dis_y(gen) + map.info.origin.position.y;

        Particle p;
        p.pose_.position.x = x;
        p.pose_.position.y = y;
        p.pose_.position.z = 0;
        p.pose_.orientation.x = 0.0;
        p.pose_.orientation.y = 0.0;
        p.pose_.orientation.z = 0.0;
        p.pose_.orientation.w = 1.0;  // Identity quaternion for zero rotation

        // Add the Particle only if it s generated on a 'non-obstacle-cell'
        if(isValid(p, map)){ // If the particle's cell it's not occupied u can add it to the sample set
            particles_.push_back(p);
        }
    }
    publishParticles(particles_);

    // 1.2 Initialize the Likelihood Field
    likelihood_.header.frame_id = "map";
    likelihood_.info = map.info;                     // Same dimension of the grid map
    likelihood_.data.resize(map.data.size(), 0);     // Initialize with 0 (free) [0-100]

    // Initializing a vecctor of distance to closest obstacle form each cell of the map 
    distanceGrid_ = computeDistanceGrid(map);        // Contains all (z_exp - z) for each cell  

    // Params - To tune
    double sigma_hit = 1.42;         // Std dev of sensor model (0.7 - 1.2)
    double z_hit_w   = 0.8;         // weight assigned to the probability of a "hit" in the sensor model
    double z_rand_w  = 1 - z_hit_w; // weight assigned to the probability of a "random" measurement
    double z_max = 30.0;            // laser_info_.range_max

    int rows = map.info.height;
    int cols = map.info.width;

    vector<vector<double>> likelihood_txt(rows, vector<double>(cols, 0));   // To store the likelihood to print on a file
    likelihood.resize(map.data.size(), 0);

    //Iterate for each cell of the map
    for(int row=0; row < rows; ++row){
        for(int col=0; col < cols; ++col){

            int idx = row * cols + col;
            int d = distanceGrid_[idx];
            // Prob. of a hit
            double p_hit = exp(-0.5 * (pow((d/sigma_hit), 2)));
            
            // Prob. of rdm measurement
            double p_rand = 1. / z_max;
            
            // Total weighted prob.
            double p_total = (z_hit_w * p_hit) + (z_rand_w * p_rand);
            //cout << "pTot: " << p_total << endl;

            // Assign it to the same cell of the map in the likelihood_ gridmap
            likelihood_txt[row][col] = p_total;
            likelihood[idx] = p_total;  
            likelihood_.data[idx] = static_cast<int8_t>(p_total * 100);  
        }
    }

    // Save the likelihod in a txt file
    // Open the file in write mode
    ofstream outputFile("Likelihood.txt");
    if(outputFile.is_open()){

        for(const auto& row : likelihood_txt){
            for(const auto& val : row){
                outputFile << val << " ";
            }
            outputFile << "\n";
        }
        outputFile.close();
    } else {
        // Output an error message if the file couldn't be opened
        cout << "Error: Unable to open the file for writing." << endl;
    }

    /*// Publish the likelihood_ map
    ros::NodeHandle lik;
    ros::Publisher occupancy_grid_pub = lik.advertise<nav_msgs::OccupancyGrid>("likelihood_heatmap", 1, true);
    occupancy_grid_pub.publish(likelihood_);
    ros::spin();
    */

    //normalize weight of particles
    normalizeParticleWeights();
}

bool LaserBasedLocalizationPf::isValid(Particle p, nav_msgs::OccupancyGrid map) {

    double x = p.getX();
    double y = p.getY();

    // obtaining idxs to access likelihood data structure
    int grid_x = static_cast<int>((x - map.info.origin.position.x) / map.info.resolution);
    int grid_y = static_cast<int>((y - map.info.origin.position.y) / map.info.resolution);
    int idx = grid_y * map.info.width + grid_x;

    if(idx < map.data.size() && map.data[idx] == 0){
        return true;
    }else{
        return false;
    }
}


// Compute the distance from each cell to the nearest obstacle (value > 100 in the gridMap) using
// the BFS algorithm
std::vector<int>  LaserBasedLocalizationPf::computeDistanceGrid(nav_msgs::OccupancyGrid map){
    int rows = map.info.height;
    int cols = map.info.width;
    const vector<pair<int, int>> directions = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    vector<int> distanceGrid(rows*cols, INF);  // Initialize with INF

    // Initialize queue with obstacle cells
    std::queue<int> q;
    for(int i=0; i<rows; ++i){
        for(int j=0; j<cols; ++j){

            int idx = i * cols + j;
            if(map.data[idx]>0){
                distanceGrid[idx] = 0;
                q.push(idx);
            }
        }
    }
    
    // BFS
    while(!q.empty()){
        int index = q.front();
        q.pop();

        int row = index / cols;
        int col = index % cols;

        // Neighbours
        for(const auto& dir : directions){
            int newRow = row + dir.first;
            int newCol = col + dir.second;

            // check if adjacent cell is valid and not visited 
            bool cond = newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols
                        && (distanceGrid[newRow * cols + newCol] == INF);

            if(cond){
                // Update distance to adjacent cell and enqueue it
                distanceGrid[newRow * cols + newCol] = distanceGrid[row * cols + col] +1;
                q.push(newRow * cols + newCol);
            }
        }
    }
    return distanceGrid;
}

void LaserBasedLocalizationPf::updateOdometry(nav_msgs::Odometry odometry){

    //todo move particles the same way robot moved
    static bool first_call = true;

    if (first_call){
        last_odometry = odometry;
        resetLocalization(odometry.pose.pose.position.x, odometry.pose.pose.position.y, tf::getYaw(odometry.pose.pose.orientation));
        updateLocalization(x,particles_);
        first_call = false;
        return;
    }


    // Checking if the robot is moving using the linear/angular velocities in order
    // to don't resample when the robot is in stop
    double linear_vel_x  = odometry.twist.twist.linear.x;
    double linear_vel_y  = odometry.twist.twist.linear.y;
    double angular_vel   = odometry.twist.twist.angular.z;
    double tresh         = 0.001;
    if(fabs(linear_vel_x) >= tresh || fabs(linear_vel_y) >= tresh || fabs(angular_vel) >= tresh){
        // Robot is moving
        is_moving = true;
      
        // 1.3 Implement the Motion Model
        // Particle's Previous Odometry:
        // Extract the new (x', y', theta')
        double x_new = odometry.pose.pose.position.x;
        double y_new = odometry.pose.pose.position.y;
        double theta_new = tf::getYaw(odometry.pose.pose.orientation);

        // To tune
        double alpha_1 = 0.01;      //How rotation affects rotation variance. (0.0, 1.0)
        double alpha_2 = 0.01;      //How translation affects rotation variance (0.0, 0.05)
        double alpha_3 = 0.01;      //How translation affects translation variance (0.0, 5.0)
        double alpha_4 = 0.01;      //How rotation affects translation variance  (0.0, 1.0)
        /*
        double alpha_1 = 0.01;      //How rotation affects rotation variance. (0.0, 1.0)
        double alpha_2 = 0.05;      //How translation affects rotation variance (0.0, 0.05)
        double alpha_3 = 1;         //How translation affects translation variance (0.0, 5.0)
        double alpha_4 = 0.01;      //How rotation affects translation variance  (0.0, 1.0)

        double alpha_1 = 0.01;      //How rotation affects rotation variance. (0.0, 1.0)
        double alpha_2 = 0.02;      //How translation affects rotation variance (0.0, 0.05)
        double alpha_3 = 3;         //How translation affects translation variance (0.0, 5.0)
        double alpha_4 = 0.25;      //How rotation affects translation variance  (0.0, 1.0)
        */

        // Robot's Previous Odometry
        // Extract the robot's previous (x, y, theta)
        double x_prev_robot = last_odometry.pose.pose.position.x;
        double y_prev_robot = last_odometry.pose.pose.position.y;
        double theta_prev_robot = tf::getYaw(last_odometry.pose.pose.orientation);

        
        // Update each particle relative to the robot's previous pose
        for(int i = 0; i < particles_.size(); i++){
            
            // Extract the particle's previous (x, y, theta)
            double x_prev_particle = particles_[i].getX();
            double y_prev_particle = particles_[i].getY();
            double theta_prev_particle = particles_[i].getTheta();

            // Compute δ_rot1, δ_trans, δ_rot2
            double delta_rot1  = atan2(y_new - y_prev_robot, x_new - x_prev_robot) - theta_prev_robot;
            double delta_rot2  = theta_new - theta_prev_robot - delta_rot1;
            double delta_trans = sqrt(pow((x_prev_robot - x_new), 2) + pow((y_prev_robot - y_new), 2)); 

            // Compute δ_rot1_hat, δ_trans_hat, δ_rot2_hat adding the noise model
            double delta_rot1_hat  = delta_rot1  + sampleNormalDistribution(alpha_1 * abs(delta_rot1)  + alpha_2 * abs(delta_trans));
            double delta_trans_hat = delta_trans + sampleNormalDistribution(alpha_3 * abs(delta_trans) + alpha_4 * abs(delta_rot1 + delta_rot2));
            double delta_rot2_hat  = delta_rot2  + sampleNormalDistribution(alpha_1 * abs(delta_rot2)  + alpha_2 * abs(delta_trans));

            // Compute the updated (x', y', theta') according to the motion model
            double x_update = x_prev_particle + delta_trans_hat * cos(theta_prev_particle + delta_rot1_hat);
            double y_update = y_prev_particle + delta_trans_hat * sin(theta_prev_particle + delta_rot1_hat);
            double theta_update = theta_prev_particle + delta_rot1_hat + delta_rot2_hat;

            // Update particle's pose
            particles_[i].updatePose(x_update, y_update, theta_update);
        }

        // global variable last_odometry contains the last odometry position estimation (ROS Odometry Messasge)
        // local variable odometry contains the current odometry position estimation (ROS Odometry Messasge)
        updateLocalization(x, particles_);
        last_odometry = odometry;
    }else{
        is_moving = false;
    }
}

void LaserBasedLocalizationPf::visualizeSeenLaser(sensor_msgs::LaserScan laser)
{
    static bool first = true;
    if(first)
        laser_info_ = laser;
    first = false;

    tf::Transform transform;
    transform.setOrigin( tf::Vector3(0.38, 0.0, 0.103) );
    transform.setRotation( tf::createQuaternionFromRPY(0, 0, 0) );
    pose_tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base_link_real", "laser_link_real"));

    laser.header.frame_id = "laser_link_real";
    real_laser_pub_.publish(laser);
}

void LaserBasedLocalizationPf::updateLaser(sensor_msgs::LaserScan laser){
    visualizeSeenLaser(laser);

    // From rviz
    double map_origin_x   = -19.5;
    double map_origin_y   = -9;
    double map_resolution = 0.05;
    double z_max = laser.range_max;

    // Since the laser scanner is not perfectly positioned
    double laser_shift = 0.38;      // From setup (robot) details 

    // Prob. settings
    double deltaZ, likelihood_val;
    double lambda = 1.1;
    double p_rand, p_unexp, p_max, tot_prob;
    double z_hit_w   = 0.60;
    double z_unexp_w = 0.05;
    double z_rand_w  = 0.30;
    double z_max_w   = 0.05;

    // 1. Compute the pose of the virutal laser for each particle
    for(int i=0; i<particles_.size(); ++i){ 
        // Reset probs for each particle
        p_rand=0, p_unexp=0, p_max=0, tot_prob=0;

        // Retrieve particle's pose
        Particle p = particles_[i];
        double theta_p = p.getTheta(); 
        // Since the laser sensor is not perfectly centered 
        double x_part  = p.getX() + laser_shift *cos(theta_p);      
        double y_part  = p.getY() + laser_shift *sin(theta_p);   

        // Get particle's posizion on the map as a cell index
        double likelihood_particle_cell = 0;
        int particle_cell_x = static_cast<int>((x_part - map_origin_x) / map_resolution);
        int particle_cell_y = static_cast<int>((y_part - map_origin_y) / map_resolution);

        // Check if the idx is whithin the borders
        if( particle_cell_x >= 0 && particle_cell_x < likelihood_.info.width &&
            particle_cell_y >= 0 && particle_cell_y < likelihood_.info.height){

            int idx = particle_cell_y * likelihood_.info.width + particle_cell_x;
            likelihood_particle_cell = likelihood_.data[idx];
        }

        // Transform the laser point to the particle frame
        try {
            tf::Transform transform;
            transform.setOrigin(tf::Vector3(x_part, y_part, 0));
            transform.setRotation(tf::createQuaternionFromYaw(theta_p));
            pose_tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), laser.header.frame_id, "particle_frame"));
        } catch (tf::TransformException &ex) {
            ROS_WARN("Failed to broadcast transform: %s", ex.what());
        }

        // 2. Derive the propbality of a laser measurement using the likelihood field
        for(int j=0; j<laser.ranges.size(); j++){

            // Convert laser measurement to Cartesian coordinates
            double angle = laser.angle_min + j * laser.angle_increment;
            double range = laser.ranges[j];
            double laser_x = range * cos(angle);    // Coords of the end-point of the ray
            double laser_y = range * sin(angle);    
            double z = laser.ranges[j];

            // obtaining idxs to access likelihood data structure
            int grid_x = static_cast<int>((laser_x - map_origin_x) / map_resolution);
            int grid_y = static_cast<int>((laser_y - map_origin_y) / map_resolution);

            // Check if the idx is whithin the borders
            if( grid_x >= 0 && grid_x < likelihood_.info.width &&
                grid_y >= 0 && grid_y < likelihood_.info.height){

                int idx = grid_y * likelihood_.info.width + grid_x;
                likelihood_val = likelihood[idx];
                double likelihood_val1 = likelihood_.data[idx];

                // (z-z_exp) computed by the distanceGrid_
                deltaZ = distanceGrid_[idx];
            
                // Computing the probabilities (Beam-Proximity model)

                if(deltaZ > 0){
                    p_rand = 1/z_max;   // P_rand
                }else if (deltaZ == 0){            
                    p_max = 1;          // P_max
                }else{
                    p_unexp = lambda * exp(-lambda * z);    //P_unexp
                }
            }else{
                p_rand = 1/z_max;
                //tot_prob += log(p_rand);    // Using log to asintotically manage extreme values
            }
            
            // Combine all togheter by weighted summation
            // Use pre-computed likelihood values to affect the weights
            // Using log to asintotically manage extreme values
            tot_prob += log((z_hit_w * likelihood_val) + (z_unexp_w * p_unexp) + (z_rand_w * p_rand) + (z_max_w * p_max));
        }   
        particles_[i].updateWeight(tot_prob);
    }
    
    // normalize your weights
    normalizeParticleWeights();
    // do resampling only when robot is moving
    if(is_moving) resamplingParticles();

    // Keep This - reports your update
    updateLocalization(x, particles_);
}

void LaserBasedLocalizationPf::resetLocalization(double x, double y, double theta){
    this->x(0,0) = x;
    this->x(1,0) = y;
    this->x(2,0) = theta;

    //distribute particles around true pose

    double scale_factor = 1000.0;

    int x_range = static_cast<int>(max_x_position_ / 10.0 * scale_factor);
    int y_range = static_cast<int>(max_y_position_ / 10.0 * scale_factor);
    int theta_range = static_cast<int>(M_PI / 4.0 * scale_factor);
    for(int i = 0; i < particles_.size(); i++)
    {
        double new_x = x + (std::rand() % x_range - static_cast<int>(x_range/2.0) ) / scale_factor;
        double new_y = y + (std::rand() % y_range - static_cast<int>(y_range/2.0) ) / scale_factor;
        double new_theta  = theta + (std::rand() % theta_range - static_cast<int>(theta_range/2.0)) / scale_factor;
        particles_[i].updatePose(new_x, new_y, new_theta);
        particles_[i].weight_ = 1.;
    }
}

void LaserBasedLocalizationPf::updateLocalization(Eigen::MatrixXd x, std::vector<Particle>& particles)
{
    //visualisation of pose
    publishPose(x, particles);

    //visualization of particles
    publishParticles(particles);
}

void LaserBasedLocalizationPf::laserCallback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
    data_mutex_->lock();
    updateLaser(*msg);
    updateLocalization(x,particles_);
    data_mutex_->unlock();
}
void LaserBasedLocalizationPf::odometryCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
    data_mutex_->lock();
    updateOdometry(*msg);
    data_mutex_->unlock();
}

void LaserBasedLocalizationPf::initialposeCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg)
{
    double x, y, theta;
    data_mutex_->lock();
    x = msg->pose.pose.position.x;
    y = msg->pose.pose.position.y;
    theta =  tf::getYaw(msg->pose.pose.orientation);
    ROS_INFO("initalPoseCallback x=%f, y=%f, theta=%f", x, y, theta);
    resetLocalization(x, y, theta);
    data_mutex_->unlock();
}

void LaserBasedLocalizationPf::mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    data_mutex_->lock();
    occ_grid_ = *msg;
    data_mutex_->unlock();
}

void LaserBasedLocalizationPf::normalizeParticleWeights(){


    // Calculate the sum of all particle weights    
    double sum_weights = 0.0;   // Normalization factor (eta)
    for(int i = 0; i < particles_.size(); ++i) {
        sum_weights += particles_[i].weight_;

        //if (particles_[i].weight_ != 0) cout << particles_[i].weight_ << " ";
    }

    // Normalize the weights_ using the sum as the normalization factor
    for(int i = 0; i < particles_.size(); ++i) {
        particles_[i].weight_ /= sum_weights;
    }

    /*double temp=0.;
    for(int i = 0; i < particles_.size(); ++i) {
        temp += particles_[i].weight_;
    }
    cout <<"Sum: "<<  sum_weights << endl;
    cout <<"Should be 1: "<<  temp << endl;   */
}

void LaserBasedLocalizationPf::resamplingParticles(){
    // The resample has to happen only if the last resampling only if the robot is moving
    // Systematic Resampling
    vector<Particle> s_prime;
    vector<double> cs;
    int n = particles_.size();

    // Computing the size of the random set of Particles to generate 
    // around the mean of the current particles
    double random_sample_perc = 0.10;
    int    random_sample_size  = static_cast<int>(random_sample_perc * n);
    //calculate mean position of given particles
    double x_mean = 0;
    double y_mean = 0;
    double t_mean = 0;

    cs.push_back(particles_[0].weight_);    // c_1 = w_1

    // Generate a starting point - Initialize threshold
    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> distribution(0.0, 1.0/n);
    double u = distribution(generator);

    // Generate PDF  
    for(int i=1; i< n - random_sample_size; ++i) {
        double c_i = cs[i-1] + particles_[i].weight_;   // c_(i-1) + w_i
        cs.push_back(c_i);  // Cumulative sum array - 
    }
    // Draw samples
    int i = 0;
    for (int j = 0; j < n - random_sample_size; ++j) {
        while (u > cs[i]) {
            ++i;
        }
        Particle p = particles_[i];

        x_mean += p.getX();
        y_mean += p.getY();
        t_mean += p.getTheta();

        // After resampling, all particles are equally likely to represent
        // the true state of the system, so they are assigned equal weights
        //p.updateWeight(1.0 / n);        // Set uniform weight
        s_prime.push_back(particles_[i]);           // Insert
        u += (1.0 / n);                 // Increment treshold
    }
    x_mean /= n;
    y_mean /= n;
    t_mean /= n;
    double neigh_size = 0.3;
    uniform_real_distribution<double> dist_x(x_mean - neigh_size, x_mean + neigh_size);
    uniform_real_distribution<double> dist_y(y_mean - neigh_size, y_mean + neigh_size);
    // Generate random samples around the mean of the position of the existing particles
    for(int i=0; i<random_sample_size; i++){
        Particle p;
        double p_x   = dist_x(generator);
        double p_y   = dist_y(generator);

        p.updatePose(p_x, p_y, t_mean);
        s_prime.push_back(p);
    }
    //cout << particles_.size() << "  -  "  << s_prime.size() << endl;
    particles_ = s_prime;
    //cout << particles_.size() << "  -  "  << s_prime.size() << endl;

}

void LaserBasedLocalizationPf::publishParticles(std::vector<Particle>& particles)
{
    geometry_msgs::PoseArray array;
    array.poses = getParticlePositions(particles);;
    array.header.frame_id = "map";
    array.header.stamp = ros::Time(0);

    particles_pub_.publish(array);
}

std::vector<geometry_msgs::Pose> LaserBasedLocalizationPf::getParticlePositions(std::vector<Particle>& particles)
{
    std::vector<geometry_msgs::Pose> positions;

    for(int i = 0; i < particles.size(); i++){
        positions.push_back(particles[i].pose_);
    }
    return positions;
}

void LaserBasedLocalizationPf::publishPose(Eigen::MatrixXd& x, std::vector<Particle>& particles)
{
    //calculate mean of given particles
    double x_mean = 0;
    double y_mean = 0;
    double yaw_mean = 0;
    int n = particles_.size();
	
	// Robot pose from the particles
    for(int i=0; i < n; i++){
        Particle p = particles_[i];

        x_mean += p.getX();
        y_mean += p.getY();
        yaw_mean += p.getTheta();
    }
    x_mean /= n;
    y_mean /= n;
    yaw_mean /= n;
	
    x(0,0) = x_mean;
    x(1,0) = y_mean;
    x(2,0) = yaw_mean;

    tf::Transform transform;
    transform.setOrigin( tf::Vector3(x_mean, y_mean, 0.0) );
    transform.setRotation( tf::createQuaternionFromRPY(0 , 0, yaw_mean) );
    pose_tf_broadcaster_.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "map", "base_link_pf"));

    //calculate covariance matrix
    double standard_deviation_x = 0;
    double standard_deviation_y = 0;
    double standard_deviation_theta = 0;

    //todo check if uncertainty possible
    for (int i = 0; i < particles.size(); i++)
    {
        standard_deviation_x += std::pow(particles[i].getX() - x_mean, 2);
        standard_deviation_y += std::pow(particles[i].getY() - y_mean, 2);
        standard_deviation_theta += std::pow(particles[i].getTheta() - yaw_mean, 2);
    }
    standard_deviation_theta = std::sqrt(standard_deviation_theta);
    standard_deviation_x = std::sqrt(standard_deviation_x);
    standard_deviation_y = std::sqrt(standard_deviation_y);
    standard_deviation_theta /= static_cast<double>(particles.size() - 1);
    standard_deviation_x /= static_cast<double>(particles.size() - 1);
    standard_deviation_y /= static_cast<double>(particles.size() - 1);

    //need to bound it otherwise calc of uncertainty marker doesn't work
    double thresh = 0.0000001;
    if(standard_deviation_theta < thresh)
        standard_deviation_theta = thresh;
    if(standard_deviation_x < thresh)
        standard_deviation_x = thresh;
    if(standard_deviation_y < thresh)
        standard_deviation_y = thresh;

    //put in right msg
    geometry_msgs::PoseWithCovarianceStamped pose_with_cov;
    pose_with_cov.header.frame_id = "map";
    pose_with_cov.header.stamp = ros::Time(0);

    tf::Quaternion q;
    q = tf::createQuaternionFromYaw(yaw_mean);

    pose_with_cov.pose.pose.position.x = x_mean;
    pose_with_cov.pose.pose.position.y = y_mean;
    pose_with_cov.pose.pose.position.z = 0;
    pose_with_cov.pose.pose.orientation.w = q.getW();
    pose_with_cov.pose.pose.orientation.x = q.getX();
    pose_with_cov.pose.pose.orientation.y = q.getY();
    pose_with_cov.pose.pose.orientation.z = q.getZ();

    pose_with_cov.pose.covariance[0] = std::pow(standard_deviation_x,2);
    pose_with_cov.pose.covariance[7] = std::pow(standard_deviation_y,2);
    pose_with_cov.pose.covariance[35] = std::pow(standard_deviation_theta,2);
    pose_with_cov_pub_.publish(pose_with_cov);

    // Uncertainty Visualization
    Eigen::Matrix2f uncertainty_mat;
    uncertainty_mat(0,0) = standard_deviation_x * 100.0;
    uncertainty_mat(0,1) = thresh;
    uncertainty_mat(1,0) = thresh;
    uncertainty_mat(1,1) = standard_deviation_y * 100.0;

    Eigen::Vector2f uncertainty_position;
    uncertainty_position(0) = x(0,0);
    uncertainty_position(1) = x(1,0);

    visualization_msgs::Marker uncertainly_marker;
    generateUncertaintyMarker(uncertainly_marker, uncertainty_mat, uncertainty_position);
    vis_pub_.publish(uncertainly_marker);
}

void LaserBasedLocalizationPf::generateUncertaintyMarker(visualization_msgs::Marker& marker, Eigen::Matrix2f uncertainly_mat, Eigen::Vector2f position)
{
    Eigen::EigenSolver<Eigen::Matrix2f> solver(uncertainly_mat);
    Eigen::VectorXf uncertainty_eigenvalues = solver.eigenvalues().real();
    //std::cout << std::endl << "Eigenvalues: " << std::endl << uncertainty_eigenvalues.transpose() << std::endl;
    Eigen::MatrixXf uncertainty_eigenvectors = solver.eigenvectors().real();
    //std::cout << std::endl << uncertainty_eigenvectors << std::endl;

    double phi_ellipse = std::atan2(uncertainty_eigenvectors(0,1), uncertainty_eigenvectors(0,0));

    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time();
    marker.ns = "ellipses";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::CYLINDER;
    marker.action = visualization_msgs::Marker::ADD;
    geometry_msgs::Pose ellipse_pose;

    ellipse_pose.position.x = position(0);
    ellipse_pose.position.y = position(1);
    ellipse_pose.position.z = 0;

    tf::Quaternion tf_quat = tf::createQuaternionFromRPY(0, 0, phi_ellipse);
    tf::quaternionTFToMsg(tf_quat, ellipse_pose.orientation);

    marker.pose = ellipse_pose;

    // eigenvalue of uncertainty matrix is the square of the semi-major/minor of the ellipse;
    // 2.447*sigma => 95% area
    marker.scale.x = 2.447*2.0*std::sqrt(uncertainty_eigenvalues(0));
    marker.scale.y = 2.447*2.0*std::sqrt(uncertainty_eigenvalues(1));
    marker.scale.z = 0.1;
    marker.color.a = 0.2;
    marker.color.r = 0.9;
    marker.color.g = 0.0;
    marker.color.b = 0.3;
}

double LaserBasedLocalizationPf::probNormalDistribution(double a, double variance)
{
    if (variance == 0)
        return a;

    return ( 1.0 / ( std::sqrt(2*M_PI * variance) ) ) * std::exp( -0.5 * std::pow( a, 2.0 ) / variance );

}

double LaserBasedLocalizationPf::sampleNormalDistribution(double variance)
{
    double scaling_factor = 1000.0;
    if (variance <= (1.0/scaling_factor))
        return 0;

    double sum = 0;

    int border = std::sqrt(variance) * static_cast<int>(scaling_factor);
    for (int i = 0; i < 12; i++)
        sum += std::rand() % (2 * border) - border;

    return sum * 0.5 / scaling_factor;

}

sensor_msgs::LaserScan::Ptr LaserBasedLocalizationPf::simulateLaser(double x, double y, double theta, double speedup)
{
    const double laser_x_dist = 0.38;
    const double laser_z_dist = 0.103;
    geometry_msgs::Pose laser_pose;
    laser_pose.position.x = x + laser_x_dist*std::cos(theta);
    laser_pose.position.y = y + laser_x_dist*std::sin(theta);
    laser_pose.position.z = laser_z_dist;
    laser_pose.orientation = tf::createQuaternionMsgFromYaw(theta);

    double inc = laser_info_.angle_increment;
    laser_info_.angle_increment = inc * speedup;
    sensor_msgs::LaserScan::Ptr simulated_laser = occupancy_grid_utils::simulateRangeScan(occ_grid_, laser_pose, laser_info_, true);
    laser_info_.angle_increment = inc;
    return simulated_laser;

}


int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "laser_based_localization");
    ros::NodeHandle n;

    LaserBasedLocalizationPf* lmbl_ptr = new LaserBasedLocalizationPf(n);


    ros::Subscriber odometry = n.subscribe("/gazebo/odom", 1, &LaserBasedLocalizationPf::odometryCallback, lmbl_ptr);
    ros::Subscriber initialpose = n.subscribe("/initialpose", 1, &LaserBasedLocalizationPf::initialposeCallback, lmbl_ptr);
    ros::Subscriber map = n.subscribe("/map", 1, &LaserBasedLocalizationPf::mapCallback, lmbl_ptr);
    ros::Subscriber laser_sub = n.subscribe("/front_laser/scan",1, &LaserBasedLocalizationPf::laserCallback, lmbl_ptr);
    //boost::thread(&Controller::stateMachine, controller);

    std::cout << "Laser Based Localization started..." << std::endl;

    ros::spin();

    return 0;
}
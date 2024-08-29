# Launching your particle filter for Ass1

- Launch Gazebo Simulation:

`roslaunch robot_gazebo platform_building.launch`

You can set the desired tf tree branch for testing your implementation by setting the `tf_base_link` argument: `base_link` sets the robot onto the odometry tf (default), `base_link_real` sets the robot onto the ground truth, and `base_link_pf` sets the odom onto your particle filter. To see how the robot behaves with your particle filter implementation, launch gazebo including the `tf_base_link` argument:
  
`roslaunch robot_gazebo platform_building.launch tf_base_link:=base_link_pf`

- Start your particle filter implementation:

`rosrun laser_based_localization_pf laser_based_localization_pf`

- Launch RViz for ROS visualization:

`roslaunch robot_gazebo rviz.launch`

- Launch Keyboard Control for moving around ([Check here for control keys](http://wiki.ros.org/teleop_twist_keyboard)):

`roslaunch robot_control teleop_keyboard.launch`


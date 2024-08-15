std_msgs/Header header

int32 instance_id

string name
int32 class_id

# image coordinate [x, y, w, h]
float32[] body_location
Pose pose

# sensor corrdinate
float32[] body_location_sensor
Pose pose_sensor

# global coordinate [X, Y, Z]
float32[] body_location_global
Pose pose_global

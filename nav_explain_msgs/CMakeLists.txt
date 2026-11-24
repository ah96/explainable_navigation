cmake_minimum_required(VERSION 3.20)
project(nav_explain_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Explanation.msg"
  "msg/ObjectAffordance.msg"
)

ament_package()

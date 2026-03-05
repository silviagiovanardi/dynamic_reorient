/*
 * Gazebo Classic world plugin for grasp attach/detach.
 *
 * Exposes two ROS 2 services:
 *   /grasp_attach  (std_srvs/SetBool)  — find nearest non-static model to
 *                                         the robot's palm link and create
 *                                         a fixed joint.
 *   /grasp_detach  (std_srvs/SetBool)  — remove the fixed joint.
 *
 * SDF parameters (inside <plugin>):
 *   <robot_model>ur5</robot_model>
 *   <robot_link>gripper_base_link</robot_link>
 *   <search_radius>0.15</search_radius>
 */

#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/set_bool.hpp>

#include <mutex>
#include <string>

namespace gazebo
{

class GazeboGraspFix : public WorldPlugin
{
public:
  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;

    if (sdf->HasElement("robot_model"))
      robot_model_name_ = sdf->Get<std::string>("robot_model");
    if (sdf->HasElement("robot_link"))
      robot_link_name_ = sdf->Get<std::string>("robot_link");
    if (sdf->HasElement("search_radius"))
      search_radius_ = sdf->Get<double>("search_radius");

    ros_node_ = gazebo_ros::Node::Get(sdf);

    attach_srv_ = ros_node_->create_service<std_srvs::srv::SetBool>(
      "/grasp_attach",
      [this](const std_srvs::srv::SetBool::Request::SharedPtr,
             std_srvs::srv::SetBool::Response::SharedPtr resp) {
        this->OnAttach(resp);
      });

    detach_srv_ = ros_node_->create_service<std_srvs::srv::SetBool>(
      "/grasp_detach",
      [this](const std_srvs::srv::SetBool::Request::SharedPtr,
             std_srvs::srv::SetBool::Response::SharedPtr resp) {
        this->OnDetach(resp);
      });

    RCLCPP_INFO(ros_node_->get_logger(),
      "GazeboGraspFix loaded: robot=%s link=%s radius=%.3f",
      robot_model_name_.c_str(), robot_link_name_.c_str(), search_radius_);
  }

private:
  // Find the robot link by name, trying multiple naming patterns
  physics::LinkPtr FindRobotLink(physics::ModelPtr robot)
  {
    // Try direct lookup first
    auto link = robot->GetLink(robot_link_name_);
    if (link) return link;

    // Try scoped name: model_name::link_name
    link = robot->GetLink(robot_model_name_ + "::" + robot_link_name_);
    if (link) return link;

    // Search all links for a name containing the target
    for (const auto &l : robot->GetLinks()) {
      std::string name = l->GetName();
      if (name.find(robot_link_name_) != std::string::npos) {
        RCLCPP_INFO(ros_node_->get_logger(),
          "Found link by partial match: %s", name.c_str());
        return l;
      }
    }

    // Log all available links for debugging
    RCLCPP_ERROR(ros_node_->get_logger(),
      "Link '%s' not found. Available links in model '%s':",
      robot_link_name_.c_str(), robot_model_name_.c_str());
    for (const auto &l : robot->GetLinks()) {
      RCLCPP_ERROR(ros_node_->get_logger(), "  - %s", l->GetName().c_str());
    }

    return nullptr;
  }

  void OnAttach(std_srvs::srv::SetBool::Response::SharedPtr resp)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    if (joint_) {
      resp->success = true;
      resp->message = "Already attached to: " + attached_model_name_;
      return;
    }

    auto robot = world_->ModelByName(robot_model_name_);
    if (!robot) {
      // Log all available models for debugging
      std::string models_list;
      for (const auto &m : world_->Models()) {
        models_list += " " + m->GetName();
      }
      resp->success = false;
      resp->message = "Robot model not found: " + robot_model_name_
                      + " (available:" + models_list + ")";
      RCLCPP_ERROR(ros_node_->get_logger(), "%s", resp->message.c_str());
      return;
    }

    auto palm = FindRobotLink(robot);
    if (!palm) {
      resp->success = false;
      resp->message = "Robot link not found: " + robot_link_name_;
      RCLCPP_ERROR(ros_node_->get_logger(), "%s", resp->message.c_str());
      return;
    }

    auto palm_pos = palm->WorldPose().Pos();
    double best_dist = search_radius_;
    physics::ModelPtr best_model;
    physics::LinkPtr  best_link;

    for (const auto &model : world_->Models()) {
      if (model->GetName() == robot_model_name_) continue;
      if (model->IsStatic()) continue;
      if (model->GetName() == "ground_plane") continue;

      for (const auto &link : model->GetLinks()) {
        double d = (link->WorldPose().Pos() - palm_pos).Length();
        if (d < best_dist) {
          best_dist = d;
          best_model = model;
          best_link  = link;
        }
      }
    }

    if (!best_model) {
      resp->success = false;
      resp->message = "No graspable model within radius";
      RCLCPP_WARN(ros_node_->get_logger(),
        "%s (palm at %.3f, %.3f, %.3f, radius=%.3f)",
        resp->message.c_str(),
        palm_pos.X(), palm_pos.Y(), palm_pos.Z(), search_radius_);
      return;
    }

    // Create a fixed joint between palm and the object link
    joint_ = world_->Physics()->CreateJoint("fixed", robot);
    joint_->SetName("grasp_fix_joint");
    joint_->Load(palm, best_link, ignition::math::Pose3d());
    joint_->Init();

    attached_model_name_ = best_model->GetName();

    resp->success = true;
    resp->message = "Attached: " + attached_model_name_;
    RCLCPP_INFO(ros_node_->get_logger(), "Attached to %s (dist=%.4f)",
      attached_model_name_.c_str(), best_dist);
  }

  void OnDetach(std_srvs::srv::SetBool::Response::SharedPtr resp)
  {
    std::lock_guard<std::mutex> lock(mtx_);

    if (!joint_) {
      resp->success = true;
      resp->message = "Nothing attached";
      return;
    }

    joint_->Detach();
    joint_.reset();

    resp->success = true;
    resp->message = "Detached: " + attached_model_name_;
    RCLCPP_INFO(ros_node_->get_logger(), "Detached %s",
      attached_model_name_.c_str());
    attached_model_name_.clear();
  }

  physics::WorldPtr world_;
  gazebo_ros::Node::SharedPtr ros_node_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr attach_srv_;
  rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr detach_srv_;
  physics::JointPtr joint_;
  std::string attached_model_name_;

  std::string robot_model_name_ = "ur5";
  std::string robot_link_name_  = "wrist_3_link";
  double search_radius_ = 0.20;

  std::mutex mtx_;
};

GZ_REGISTER_WORLD_PLUGIN(GazeboGraspFix)

}  // namespace gazebo

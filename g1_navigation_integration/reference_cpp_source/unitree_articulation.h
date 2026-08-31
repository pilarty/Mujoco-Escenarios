// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/assets/articulation/articulation.h"

namespace unitree
{

template <typename LowStatePtr>
class BaseArticulation : public isaaclab::Articulation
{
public:
    BaseArticulation(LowStatePtr lowstate_)
    : lowstate(lowstate_)
    {
        data.joystick = &lowstate->joystick;
        // NOTE: simulation-only pose source (see articulation.h). Real hardware
        // must replace this with a subscription to the SLAM/localization module.
        sportmode_state_ = std::make_shared<unitree::robot::go2::subscription::SportModeState>();
    }

    void update() override
    {
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        // base_angular_velocity
        for(int i(0); i<3; i++) {
            data.root_ang_vel_b[i] = lowstate->msg_.imu_state().gyroscope()[i];
        }
        // project_gravity_body
        data.root_quat_w = Eigen::Quaternionf(
            lowstate->msg_.imu_state().quaternion()[0],
            lowstate->msg_.imu_state().quaternion()[1],
            lowstate->msg_.imu_state().quaternion()[2],
            lowstate->msg_.imu_state().quaternion()[3]
        );
        data.projected_gravity_b = data.root_quat_w.conjugate() * data.GRAVITY_VEC_W;
        // joint positions and velocities
        for(int i(0); i< data.joint_ids_map.size(); i++) {
            data.joint_pos[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].q();
            data.joint_vel[i] = lowstate->msg_.motor_state()[data.joint_ids_map[i]].dq();
        }
        // world pose (simulation-only, see NOTE in constructor)
        if(sportmode_state_ && !sportmode_state_->isTimeout()) {
            std::lock_guard<std::mutex> sm_lock(sportmode_state_->mutex_);
            for(int i(0); i<3; i++) {
                data.root_pos_w[i] = sportmode_state_->msg_.position()[i];
                data.root_lin_vel_w[i] = sportmode_state_->msg_.velocity()[i];
            }
        }
    }

    LowStatePtr lowstate;
    std::shared_ptr<unitree::robot::go2::subscription::SportModeState> sportmode_state_;
};

}
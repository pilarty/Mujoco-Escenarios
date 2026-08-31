#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "NavigationState.h"
#include <unordered_map>

namespace isaaclab
{
// keyboard velocity commands example
// change "velocity_commands" observation name in policy deploy.yaml to "keyboard_velocity_commands"
REGISTER_OBSERVATION(keyboard_velocity_commands)
{
    std::string key = FSMState::keyboard->key();
    static auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    static std::unordered_map<std::string, std::vector<float>> key_commands = {
        {"w", {1.0f, 0.0f, 0.0f}},
        {"s", {-1.0f, 0.0f, 0.0f}},
        {"a", {0.0f, 1.0f, 0.0f}},
        {"d", {0.0f, -1.0f, 0.0f}},
        {"q", {0.0f, 0.0f, 1.0f}},
        {"e", {0.0f, 0.0f, -1.0f}}
    };
    std::vector<float> cmd = {0.0f, 0.0f, 0.0f};
    if (key_commands.find(key) != key_commands.end())
    {
        cmd = key_commands[key];
    }
    return cmd;
}

// navigation velocity commands: computes vx, vy, yaw_rate towards a hardcoded
// (x, y) goal in world frame, using root_pos_w / root_quat_w (simulation-only
// pose source, see NOTE in unitree_articulation.h / articulation.h).
REGISTER_OBSERVATION(navigation_velocity_commands)
{
    auto & data = env->robot->data;
    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    // Load the waypoint list once from config.yaml (FSM.Velocity.waypoints).
    // Falls back to a single goal from deploy.yaml's goal_x/goal_y params if
    // no waypoints are configured, preserving the single-goal behavior.
    if (!g_nav_state.loaded)
    {
        auto wp_node = param::config["FSM"]["Velocity"]["waypoints"];
        if (wp_node)
        {
            auto raw = wp_node.as<std::vector<std::vector<float>>>();
            for (auto & p : raw) g_nav_state.waypoints.push_back({p[0], p[1]});
        }
        if (g_nav_state.waypoints.empty())
        {
            float fx = params["goal_x"] ? params["goal_x"].as<float>() : 0.0f;
            float fy = params["goal_y"] ? params["goal_y"].as<float>() : 0.0f;
            g_nav_state.waypoints.push_back({fx, fy});
        }
        g_nav_state.loaded = true;
    }
    size_t idx = std::min(g_nav_state.current_idx, g_nav_state.waypoints.size() - 1);
    float goal_x = g_nav_state.waypoints[idx].first;
    float goal_y = g_nav_state.waypoints[idx].second;
    float kp_lin = params["kp_lin"] ? params["kp_lin"].as<float>() : 1.0f;
    float kp_ang = params["kp_ang"] ? params["kp_ang"].as<float>() : 2.0f;
    float goal_tolerance = params["goal_tolerance"] ? params["goal_tolerance"].as<float>() : 0.2f;

    float slow_radius = params["slow_radius"] ? params["slow_radius"].as<float>() : 0.6f;
    float heading_deadband = params["heading_deadband"] ? params["heading_deadband"].as<float>() : 0.3f;

    float dx = goal_x - data.root_pos_w[0];
    float dy = goal_y - data.root_pos_w[1];
    float distance = std::sqrt(dx*dx + dy*dy);
    g_nav_state.last_distance = distance;
    g_nav_state.last_speed = std::sqrt(
        data.root_lin_vel_w[0]*data.root_lin_vel_w[0] +
        data.root_lin_vel_w[1]*data.root_lin_vel_w[1]
    );

    float qw = data.root_quat_w.w();
    float qx = data.root_quat_w.x();
    float qy = data.root_quat_w.y();
    float qz = data.root_quat_w.z();
    float yaw = std::atan2(2.0f*(qw*qz + qx*qy), 1.0f - 2.0f*(qy*qy + qz*qz));

    float vx_body =  std::cos(yaw)*dx + std::sin(yaw)*dy;
    float vy_body = -std::sin(yaw)*dx + std::cos(yaw)*dy;

    // "arrival" style speed ramp: full gain outside slow_radius, linearly
    // scaled down to 0 as distance shrinks towards goal_tolerance. Avoids
    // the bang-bang saturation of pure proportional control at long range.
    float ramp = 1.0f;
    if (slow_radius > goal_tolerance)
    {
        ramp = std::clamp((distance - goal_tolerance) / (slow_radius - goal_tolerance), 0.0f, 1.0f);
    }

    // vy_gain scales the lateral (strafe) command. Default 1.0 = original
    // holonomic behavior (vx+vy+yaw all commanded simultaneously). Set to 0.0
    // in deploy.yaml to disable strafing entirely: the robot then only turns
    // (yaw_rate) to face the goal and walks straight (vx), closer to how a
    // human drives it with a joystick (rarely combines strong forward+strafe+
    // turn at once). TO REVERT: set vy_gain back to 1.0 in deploy.yaml, no
    // code change needed.
    float vy_gain = params["vy_gain"] ? params["vy_gain"].as<float>() : 1.0f;

    // Multi-waypoint pause/advance, done entirely within Velocity (no FSM
    // state change). Earlier version handed off to FixStand while paused,
    // but FixStand's rigid PD interpolation has no balance control and was
    // confirmed (repeatedly, at all 3 waypoints) to make the robot fall if
    // caught mid-stride. Staying in Velocity with cmd=(0,0,0) keeps the
    // walking policy (which DOES balance) in control the whole time - it
    // steps in place instead of standing perfectly still, but never falls.
    // TO REVERT to the FixStand handoff approach: see git history / the
    // State_RLBase.cpp.bak9 backup made just before this change.
    float pause_duration = param::config["FSM"]["Velocity"]["pause_duration"] ?
        param::config["FSM"]["Velocity"]["pause_duration"].as<float>() : 3.0f;
    // arrival_radius (looser than goal_tolerance) gates the pause/advance
    // logic below. goal_tolerance alone plateaus asymptotically (the ramp
    // slows cmd to near-zero as distance nears it, but residual walking
    // dynamics keep distance hovering just above goal_tolerance without
    // ever crossing it) - confirmed empirically (robot stuck at dist~0.29
    // against goal_tolerance=0.2, never advancing). arrival_radius=0.4 is
    // comfortably past that plateau.
    float arrival_radius = cfg["arrival_radius"] ? cfg["arrival_radius"].as<float>() : 0.4f;
    bool waypoints_remaining = (g_nav_state.current_idx + 1 < g_nav_state.waypoints.size());

    std::vector<float> cmd(3, 0.0f);
    if (distance <= arrival_radius)
    {
        if (!g_nav_state.waiting_at_waypoint)
        {
            g_nav_state.waiting_at_waypoint = true;
            g_nav_state.arrived_time = std::chrono::steady_clock::now();
        }
        else if (waypoints_remaining)
        {
            float elapsed = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - g_nav_state.arrived_time).count();
            if (elapsed >= pause_duration)
            {
                g_nav_state.current_idx++;
                g_nav_state.waiting_at_waypoint = false;
            }
        }
        // cmd stays (0,0,0): either paused at a waypoint, or this was the
        // last one and there's nothing further to walk to.
    }
    else
    {
        g_nav_state.waiting_at_waypoint = false;
        cmd[0] = std::clamp(kp_lin * vx_body * ramp, cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
        cmd[1] = std::clamp(kp_lin * vy_body * ramp * vy_gain, cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());

        // heading correction is skipped inside the deadband: atan2(dy, dx) is
        // numerically unstable at short range (tiny position noise causes huge
        // target-heading swings), which was the source of the erratic spinning
        // observed near the goal.
        if (distance > heading_deadband)
        {
            float target_yaw = std::atan2(dy, dx);
            float yaw_error = target_yaw - yaw;
            while (yaw_error > M_PI) yaw_error -= 2.0f*M_PI;
            while (yaw_error < -M_PI) yaw_error += 2.0f*M_PI;
            cmd[2] = std::clamp(kp_ang * yaw_error, cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
        }
    }

    // ---------------------------------------------------------------------
    // EXPERIMENTAL - added to address observed instability (violent yaw_rate
    // sign flips and real IMU yaw oscillation >180 deg/s far from the goal,
    // suspected feedback loop between the 50Hz proportional recompute and the
    // walking policy's response lag). Exponential smoothing on the final cmd
    // before it's returned/consumed as the policy's velocity_commands input.
    // TO REVERT: delete this block (from the "EXPERIMENTAL" comment down to
    // the matching "END EXPERIMENTAL" comment) and nothing else needs to
    // change; cmd will flow through as computed above, unsmoothed.
    {
        float smoothing_alpha = params["cmd_smoothing_alpha"] ? params["cmd_smoothing_alpha"].as<float>() : 0.15f;
        static std::vector<float> smoothed_cmd(3, 0.0f);
        for (int i = 0; i < 3; ++i)
        {
            smoothed_cmd[i] = smoothing_alpha * cmd[i] + (1.0f - smoothing_alpha) * smoothed_cmd[i];
        }
        cmd = smoothed_cmd;
    }
    // END EXPERIMENTAL
    // ---------------------------------------------------------------------

    // throttled debug log (~1 Hz at step_dt=0.02s => 50 calls/s)
    static int nav_log_counter = 0;
    if (nav_log_counter++ % 50 == 0)
    {
        spdlog::info(
            "[nav] pos=({:.2f},{:.2f}) yaw={:.2f} goal=({:.2f},{:.2f}) dist={:.2f} cmd=({:.2f},{:.2f},{:.2f})",
            data.root_pos_w[0], data.root_pos_w[1], yaw, goal_x, goal_y, distance, cmd[0], cmd[1], cmd[2]
        );
    }

    return cmd;
}

// NOTE: mirrors isaaclab::mdp::gait_phase (observations.h), but calls
// navigation_velocity_commands instead of the hardcoded joystick-based
// velocity_commands. gait_phase's cmd_norm check (freezing the phase signal
// when commanded velocity is near zero) was silently always seeing ~0 from
// the joystick, since our ScriptedJoystick never sets lx/ly/rx - even while
// the robot was actually being commanded to walk via navigation. Suspected
// cause of the erratic/unstable gait observed. TO REVERT: switch deploy.yaml
// back to "gait_phase" instead of "navigation_gait_phase".
REGISTER_OBSERVATION(navigation_gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    auto cmd = isaaclab::navigation_velocity_commands(env, params);
    float cmd_norm = std::sqrt(
        cmd[0] * cmd[0] +
        cmd[1] * cmd[1] +
        cmd[2] * cmd[2]
    );

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);

    if (cmd_norm < 0.1f)
    {
        obs[0] = 0.0f;
        obs[1] = 0.0f;
    }

    return obs;
}

}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

}

void State_RLBase::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
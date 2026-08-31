// Shared state for chaining multiple navigation waypoints with a pause at
// each one. Coordinates between State_RLBase (Velocity: walks toward the
// current waypoint, detects arrival) and State_FixStand (holds still during
// the pause, then advances to the next waypoint and returns to Velocity).
#pragma once
#include <vector>
#include <utility>
#include <chrono>
#include <cstddef>

struct NavigationState
{
    std::vector<std::pair<float,float>> waypoints;
    size_t current_idx = 0;
    bool loaded = false;

    // Updated every navigation_velocity_commands() call (~50Hz policy rate).
    float last_distance = 1e9f;

    // Horizontal speed (m/s), from root_lin_vel_w. Used together with
    // last_distance so the Velocity->FixStand handoff only happens once the
    // robot has actually slowed down, not mid-stride - FixStand uses rigid
    // PD interpolation with no balance control, so transferring control
    // while the robot still has forward momentum causes it to fall.
    float last_speed = 1e9f;

    // True while paused at a waypoint (set by Velocity's arrival check,
    // cleared by FixStand once it advances to the next waypoint). Lets
    // FixStand tell apart "paused because we arrived" from its original
    // use (initial get-up from Passive at startup), which must NOT
    // auto-advance.
    bool waiting_at_waypoint = false;
    std::chrono::steady_clock::time_point arrived_time;
};

inline NavigationState g_nav_state;

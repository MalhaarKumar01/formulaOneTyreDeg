#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>

// CAN frame IDs — F1 telemetry channel assignments
static constexpr uint32_t CAN_ID_TYRE_THERMAL  = 0x001;  // 4 tyres × 3 zones thermal array
static constexpr uint32_t CAN_ID_FLUID_DYNAMICS = 0x002;  // brake pressure, oil/coolant/fuel
static constexpr uint32_t CAN_ID_KINEMATICS     = 0x003;  // speed, throttle, RPM, DRS

// Raw CAN 2.0B frame (11-bit ID, 8-byte DLC — standard automotive spec)
struct CanFrame {
    uint32_t id;
    uint8_t  dlc;          // data length code (max 8)
    uint8_t  data[8];
    uint64_t timestamp_ns; // monotonic nanoseconds at capture
};

// ---------------------------------------------------------------------------
// Parsed domain structs — these live directly in ring buffer slots (zero-copy)
// ---------------------------------------------------------------------------

struct TyreThermalMatrix {
    // [tyre][zone]: tyre order = FL, FR, RL, RR; zone = inner, mid, outer
    float temp[4][3];

    float avg_surface() const {
        float s = 0.f;
        for (int t = 0; t < 4; ++t)
            for (int z = 0; z < 3; ++z)
                s += temp[t][z];
        return s / 12.f;
    }
    float max_delta() const {
        float mn = temp[0][0], mx = temp[0][0];
        for (int t = 0; t < 4; ++t)
            for (int z = 0; z < 3; ++z) {
                mn = std::min(mn, temp[t][z]);
                mx = std::max(mx, temp[t][z]);
            }
        return mx - mn;
    }
};

struct FluidDynamics {
    float brake_pressure_bar;
    float oil_temp_c;
    float coolant_temp_c;
    float fuel_pressure_bar;
};

struct KinematicsFrame {
    float speed_kph;
    float throttle_pct;   // 0–100
    float rpm;
    float drs_active;     // 0 or 1
};

// Aggregated lap-level struct — what MQTT publishes (one per lap cycle)
struct LapTelemetry {
    uint32_t lap_number;
    uint8_t  compound;           // 0=HARD 1=INTER 2=MEDIUM 3=SOFT 4=WET
    uint8_t  stint_lap;          // lap within current stint
    float    avg_tyre_temp;
    float    max_tyre_delta;
    float    avg_throttle_pct;
    float    full_throttle_pct;  // fraction of lap at ≥98% throttle
    float    avg_brake;
    float    braking_pct;
    float    max_speed_kph;
    float    drs_active_pct;
    float    avg_rpm;
    float    track_temp;
    float    air_temp;
    float    fuel_load_kg;
};

// ---------------------------------------------------------------------------
// CAN frame encoder/decoder helpers
// ---------------------------------------------------------------------------

inline void encode_float_pair(uint8_t* dst, float a, float b) {
    int16_t ia = static_cast<int16_t>(a * 10.f);
    int16_t ib = static_cast<int16_t>(b * 10.f);
    std::memcpy(dst + 0, &ia, 2);
    std::memcpy(dst + 2, &ib, 2);
}

inline void decode_float_pair(const uint8_t* src, float& a, float& b) {
    int16_t ia, ib;
    std::memcpy(&ia, src + 0, 2);
    std::memcpy(&ib, src + 2, 2);
    a = ia / 10.f;
    b = ib / 10.f;
}

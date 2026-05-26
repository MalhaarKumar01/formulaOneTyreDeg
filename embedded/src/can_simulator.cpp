#include "../include/can_simulator.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>

static uint64_t mono_ns() {
    return static_cast<uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch() /
        std::chrono::nanoseconds(1));
}

static std::mt19937& rng_instance() {
    static std::mt19937 g{42};
    return g;
}

static float randf(float lo, float hi) {
    return std::uniform_real_distribution<float>{lo, hi}(rng_instance());
}

static float randnorm(float mean, float sd) {
    return std::normal_distribution<float>{mean, sd}(rng_instance());
}

CanSimulator::CanSimulator(uint8_t compound, float starting_fuel_kg)
    : compound_(compound), fuel_kg_(starting_fuel_kg), stint_lap_(0), tick_(0)
{
    float base = 80.f + static_cast<float>(3 - std::min<int>(compound, 3)) * 5.f;
    for (int t = 0; t < 4; ++t)
        for (int z = 0; z < 3; ++z)
            tyre_temp_[t][z] = base + randf(-5.f, 5.f);
}

void CanSimulator::tick() {
    ++tick_;
    if (tick_ % 90 == 0) {
        ++stint_lap_;
        fuel_kg_ = std::max(0.f, fuel_kg_ - 1.88f);
    }
    float heat = (stint_lap_ < 3) ? 0.5f : -0.05f;
    for (int t = 0; t < 4; ++t)
        for (int z = 0; z < 3; ++z) {
            float v = tyre_temp_[t][z] + heat + randnorm(0.f, 0.4f);
            tyre_temp_[t][z] = std::max(60.f, std::min(130.f, v));
        }
}

static void pack_i16(uint8_t* dst, int offset, int16_t val) {
    std::memcpy(dst + offset * 2, &val, 2);
}

static int16_t enc10(float v)  { return static_cast<int16_t>(v * 10.f); }
static int16_t enc100(float v) { return static_cast<int16_t>(v * 100.f); }

CanFrame CanSimulator::make_tyre_frame(int half) const {
    CanFrame f{};
    f.id           = CAN_ID_TYRE_THERMAL;
    f.dlc          = 8;
    f.timestamp_ns = mono_ns();

    if (half == 0) {
        pack_i16(f.data, 0, enc10(tyre_temp_[0][0])); // FL inner
        pack_i16(f.data, 1, enc10(tyre_temp_[0][1])); // FL mid
        pack_i16(f.data, 2, enc10(tyre_temp_[0][2])); // FL outer
        pack_i16(f.data, 3, enc10(tyre_temp_[1][0])); // FR inner
    } else {
        pack_i16(f.data, 0, enc10(tyre_temp_[1][1])); // FR mid
        pack_i16(f.data, 1, enc10(tyre_temp_[1][2])); // FR outer
        pack_i16(f.data, 2, enc10(tyre_temp_[2][1])); // RL mid
        pack_i16(f.data, 3, enc10(tyre_temp_[3][1])); // RR mid
    }
    return f;
}

CanFrame CanSimulator::make_fluid_frame() const {
    CanFrame f{};
    f.id           = CAN_ID_FLUID_DYNAMICS;
    f.dlc          = 8;
    f.timestamp_ns = mono_ns();

    pack_i16(f.data, 0, enc100(randf(30.f, 120.f)));      // brake pressure bar
    pack_i16(f.data, 1, enc10(114.f + randf(0.f, 5.f)));  // oil temp °C
    pack_i16(f.data, 2, enc10(87.f  + randf(0.f, 4.f)));  // coolant temp °C
    pack_i16(f.data, 3, enc100(3.8f + randf(0.f, 0.1f))); // fuel pressure bar
    return f;
}

CanFrame CanSimulator::make_kinematics_frame() const {
    CanFrame f{};
    f.id           = CAN_ID_KINEMATICS;
    f.dlc          = 8;
    f.timestamp_ns = mono_ns();

    float spd = randf(80.f, 330.f);
    float thr = randf(60.f, 100.f);
    float rpm = randf(8000.f, 15000.f);
    float drs = (spd > 250.f) ? 1.f : 0.f;

    pack_i16(f.data, 0, static_cast<int16_t>(spd));
    pack_i16(f.data, 1, enc10(thr));
    pack_i16(f.data, 2, static_cast<int16_t>(rpm));
    pack_i16(f.data, 3, static_cast<int16_t>(drs));
    return f;
}

float   CanSimulator::fuel_load()  const { return fuel_kg_;   }
uint8_t CanSimulator::stint_lap()  const { return stint_lap_; }
uint8_t CanSimulator::compound()   const { return compound_;  }

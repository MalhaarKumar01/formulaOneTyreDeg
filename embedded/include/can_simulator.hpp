#pragma once
#include "can_frame.hpp"
#include <cstdint>

class CanSimulator {
public:
    explicit CanSimulator(uint8_t compound = 2, float starting_fuel_kg = 110.f);

    void     tick();                        // advance 10ms simulation step
    CanFrame make_tyre_frame(int half) const;
    CanFrame make_fluid_frame() const;
    CanFrame make_kinematics_frame() const;

    float   fuel_load()  const;
    uint8_t stint_lap()  const;
    uint8_t compound()   const;

private:
    uint8_t compound_;
    float   fuel_kg_;
    uint8_t stint_lap_;
    uint32_t tick_;
    float   tyre_temp_[4][3];
};

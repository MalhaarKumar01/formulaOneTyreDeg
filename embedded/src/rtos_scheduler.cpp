// F1 Embedded Telemetry Core — main entry point.
//
// Wires together:
//   CAN simulator → ring buffers (thermal, fluid, kinematics)
//   → priority-driven task scheduler (HIGH parse / MED aggregate / LOW MQTT)
//   → MQTT broker on localhost:1883
//
// Run alongside the FastAPI server + mqtt_consumer.py bridge.

#include "../include/can_frame.hpp"
#include "../include/can_simulator.hpp"
#include "../include/mqtt_publisher.hpp"
#include "../include/ring_buffer.hpp"
#include "../include/task_scheduler.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <thread>

static volatile bool g_running = true;

static void on_signal(int) { g_running = false; }

// Aggregate several tyre frames into a single lap-level summary
static LapTelemetry build_lap_telemetry(const CanSimulator& sim,
                                         float avg_temp, float max_delta,
                                         float avg_thr, float full_thr,
                                         float avg_brk, float brk_pct,
                                         float max_spd, float drs_pct,
                                         float avg_rpm)
{
    LapTelemetry lt{};
    lt.lap_number      = sim.stint_lap();
    lt.compound        = sim.compound();
    lt.stint_lap       = sim.stint_lap();
    lt.avg_tyre_temp   = avg_temp;
    lt.max_tyre_delta  = max_delta;
    lt.avg_throttle_pct = avg_thr;
    lt.full_throttle_pct = full_thr;
    lt.avg_brake       = avg_brk;
    lt.braking_pct     = brk_pct;
    lt.max_speed_kph   = max_spd;
    lt.drs_active_pct  = drs_pct;
    lt.avg_rpm         = avg_rpm;
    lt.track_temp      = 38.f;   // static mock — would come from weather CAN channel
    lt.air_temp        = 26.f;
    lt.fuel_load_kg    = sim.fuel_load();
    return lt;
}

// Format LapTelemetry as JSON for MQTT payload
static int format_json(char* buf, int bufsz, const LapTelemetry& lt) {
    static const char* compounds[] = {"HARD","INTERMEDIATE","MEDIUM","SOFT","WET"};
    const char* cmp = lt.compound < 5 ? compounds[lt.compound] : "UNKNOWN";
    return std::snprintf(buf, bufsz,
        "{"
        "\"lap\":%u,\"compound\":\"%s\",\"stint_lap\":%u,"
        "\"avg_tyre_temp\":%.1f,\"max_tyre_delta\":%.1f,"
        "\"avg_throttle_pct\":%.1f,\"full_throttle_pct\":%.3f,"
        "\"avg_brake\":%.3f,\"braking_pct\":%.3f,"
        "\"max_speed_kph\":%.1f,\"drs_active_pct\":%.3f,"
        "\"avg_rpm\":%.0f,\"track_temp\":%.1f,\"air_temp\":%.1f,"
        "\"fuel_load_kg\":%.1f"
        "}",
        lt.lap_number, cmp, lt.stint_lap,
        lt.avg_tyre_temp, lt.max_tyre_delta,
        lt.avg_throttle_pct, lt.full_throttle_pct,
        lt.avg_brake, lt.braking_pct,
        lt.max_speed_kph, lt.drs_active_pct,
        lt.avg_rpm, lt.track_temp, lt.air_temp,
        lt.fuel_load_kg);
}

int main() {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    std::printf("[RTOS] F1 Embedded Telemetry Core starting...\n");

    // Ring buffers — mmap-backed, SPSC, power-of-2 capacity
    RingBuffer<CanFrame, 256> thermal_buf;
    RingBuffer<CanFrame, 256> fluid_buf;
    RingBuffer<CanFrame, 256> kin_buf;

    // Priority scheduler — 3 worker threads
    TaskScheduler scheduler(3);

    // MQTT publisher
    MqttPublisher mqtt;
    if (!mqtt.connected()) {
        std::fprintf(stderr, "[RTOS] MQTT broker not available — run: brew services start mosquitto\n");
        std::fprintf(stderr, "[RTOS] Continuing without MQTT (frames still parsed + ring-buffered)\n");
    }

    // CAN simulator: SOFT tyres, 110kg fuel
    CanSimulator sim(3 /*SOFT*/, 110.f);

    // Lap-level accumulators (reset each simulated lap ~every 90 ticks)
    float sum_temp = 0.f, max_delta = 0.f;
    float sum_thr = 0.f, sum_full_thr = 0.f;
    float sum_brk = 0.f, sum_brk_pct = 0.f;
    float max_spd = 0.f, sum_drs = 0.f, sum_rpm = 0.f;
    int   n_kin = 0;
    uint8_t last_lap = 0;

    while (g_running) {
        sim.tick();

        // ---------------------------------------------------------------
        // HIGH-priority task: interrupt handler — parse CAN frames, write
        // directly into ring buffer slots. Returns immediately, no I/O.
        // ---------------------------------------------------------------
        scheduler.submit(Priority::HIGH, [&]() {
            CanFrame tf0 = sim.make_tyre_frame(0);
            CanFrame tf1 = sim.make_tyre_frame(1);
            CanFrame ff  = sim.make_fluid_frame();
            CanFrame kf  = sim.make_kinematics_frame();

            thermal_buf.push(tf0);
            thermal_buf.push(tf1);
            fluid_buf.push(ff);
            kin_buf.push(kf);
        });

        // ---------------------------------------------------------------
        // MED-priority task: feature aggregator — drain ring buffers,
        // compute lap-level stats.
        // ---------------------------------------------------------------
        scheduler.submit(Priority::MED, [&]() {
            TyreThermalMatrix ttm{};
            CanFrame cf{};

            // Drain thermal frames — decode temp pairs
            while (thermal_buf.pop(cf)) {
                for (int i = 0; i < 4; ++i) {
                    int16_t raw;
                    std::memcpy(&raw, cf.data + i * 2, 2);
                    ttm.temp[i / 2][i % 3] = raw / 10.f;
                }
                sum_temp  += ttm.avg_surface();
                float d    = ttm.max_delta();
                if (d > max_delta) max_delta = d;
            }

            // Drain kinematics frames
            while (kin_buf.pop(cf)) {
                int16_t spd_i, thr_i, rpm_i, drs_i;
                std::memcpy(&spd_i, cf.data + 0, 2);
                std::memcpy(&thr_i, cf.data + 2, 2);
                std::memcpy(&rpm_i, cf.data + 4, 2);
                std::memcpy(&drs_i, cf.data + 6, 2);

                float spd = static_cast<float>(spd_i);
                float thr = thr_i / 10.f;
                float rpm = static_cast<float>(rpm_i);
                float drs = static_cast<float>(drs_i);

                if (spd > max_spd) max_spd = spd;
                sum_thr      += thr;
                sum_full_thr += (thr >= 98.f) ? 1.f : 0.f;
                sum_brk      += 0.f;  // brake not in kinematics frame
                sum_brk_pct  += 0.f;
                sum_drs      += drs;
                sum_rpm      += rpm;
                ++n_kin;
            }

            // Drain fluid frames (discard — data is in JSON but not in model features)
            while (fluid_buf.pop(cf)) {}
        });

        // ---------------------------------------------------------------
        // LOW-priority task: MQTT publisher — serialize aggregated lap
        // struct and publish. This is the I/O-heavy step deliberately
        // pushed to LOW priority so it never blocks the interrupt handler.
        // ---------------------------------------------------------------
        if (sim.stint_lap() != last_lap && sim.stint_lap() > 0) {
            last_lap = sim.stint_lap();
            int n = (n_kin > 0) ? n_kin : 1;
            LapTelemetry lt = build_lap_telemetry(
                sim,
                n_kin > 0 ? sum_temp / n : 0.f,
                max_delta,
                sum_thr / n, sum_full_thr / n,
                sum_brk / n, sum_brk_pct / n,
                max_spd, sum_drs / n, sum_rpm / n);

            // Reset accumulators
            sum_temp = max_delta = sum_thr = sum_full_thr = 0.f;
            sum_brk = sum_brk_pct = max_spd = sum_drs = sum_rpm = 0.f;
            n_kin = 0;

            scheduler.submit(Priority::LOW, [lt, &mqtt]() {
                char json[512];
                int  len = format_json(json, sizeof(json), lt);
                if (len > 0)
                    mqtt.publish("f1/telemetry/lap", json, len);
                std::printf("[LAP %2u | stint_lap=%u] tyre=%.1f°C fuel=%.1fkg → MQTT\n",
                            lt.lap_number, lt.stint_lap,
                            lt.avg_tyre_temp, lt.fuel_load_kg);
            });
        }

        // 10ms tick — simulates 100Hz CAN bus sampling
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::printf("\n[RTOS] Shutting down.\n");
    std::printf("[RTOS] HIGH task WCET: max=%lluµs\n",
                (unsigned long long)(scheduler.stats(Priority::HIGH).max_ns / 1000));
    std::printf("[RTOS] MED  task WCET: max=%lluµs\n",
                (unsigned long long)(scheduler.stats(Priority::MED).max_ns / 1000));
    std::printf("[RTOS] LOW  task WCET: max=%lluµs\n",
                (unsigned long long)(scheduler.stats(Priority::LOW).max_ns / 1000));
    return 0;
}

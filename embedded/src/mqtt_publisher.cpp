#include "../include/mqtt_publisher.hpp"
#include <cstdio>
#include <cstring>
#include <mosquitto.h>

MqttPublisher::MqttPublisher(const char* host, int port, const char* client_id)
    : connected_(false)
{
    mosquitto_lib_init();
    mosq_ = mosquitto_new(client_id, true, nullptr);
    if (!mosq_) {
        std::fprintf(stderr, "[MQTT] mosquitto_new failed\n");
        return;
    }
    int rc = mosquitto_connect(mosq_, host, port, 60);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::fprintf(stderr, "[MQTT] connect failed: %s\n", mosquitto_strerror(rc));
        return;
    }
    mosquitto_loop_start(mosq_);
    connected_ = true;
}

MqttPublisher::~MqttPublisher() {
    if (mosq_) {
        mosquitto_loop_stop(mosq_, true);
        mosquitto_disconnect(mosq_);
        mosquitto_destroy(mosq_);
    }
    mosquitto_lib_cleanup();
}

// Zero-copy publish: payload pointer comes directly from a ring buffer slot.
// We do not copy into a temporary buffer — mosquitto_publish copies internally
// only what it needs for the network write, and the slot is immediately reusable.
bool MqttPublisher::publish(const char* topic, const void* payload, int len, int qos) {
    if (!connected_ || !mosq_) return false;
    int rc = mosquitto_publish(mosq_, nullptr, topic, len,
                               payload, qos, /*retain=*/false);
    if (rc != MOSQ_ERR_SUCCESS) {
        std::fprintf(stderr, "[MQTT] publish failed on %s: %s\n",
                     topic, mosquitto_strerror(rc));
        return false;
    }
    return true;
}

bool MqttPublisher::connected() const { return connected_; }

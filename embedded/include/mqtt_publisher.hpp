#pragma once

struct mosquitto;

class MqttPublisher {
public:
    MqttPublisher(const char* host = "localhost", int port = 1883,
                  const char* client_id = "f1_telemetry_core");
    ~MqttPublisher();

    MqttPublisher(const MqttPublisher&) = delete;
    MqttPublisher& operator=(const MqttPublisher&) = delete;

    // Publish raw bytes directly from caller's buffer (zero intermediate copy).
    bool publish(const char* topic, const void* payload, int len, int qos = 0);
    bool connected() const;

private:
    struct mosquitto* mosq_;
    bool connected_;
};

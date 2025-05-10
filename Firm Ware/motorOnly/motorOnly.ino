#include <WiFi.h>
#include <AsyncUDP.h>

const char* ssid = "aiLite_0128";
const char* password = "12345678";
const int udpPort = 8888;  // UDP port

// Modified network configuration to work with camera
IPAddress local_IP(192, 168, 82, 10);
IPAddress gateway(192, 168, 82, 1);
IPAddress subnet(255, 255, 255, 0);

// Reserved IPs:
// 192.168.82.10 - Robot Control ESP32
// 192.168.82.14 - ESP32-CAM

#define motorpinLF1 14
#define motorpinLF2 16
#define motorpinLB1 32
#define motorpinLB2 33
#define motorpinRF1 19
#define motorpinRF2 17
#define motorpinRB1 25
#define motorpinRB2 26

const int motorEnablePin = 13;
AsyncUDP udp;

void setup() {
    Serial.begin(115200);
    
    // Configure AP with DHCP range that won't conflict with static IPs
    WiFi.softAP(ssid, password);
    WiFi.softAPConfig(local_IP, gateway, subnet);
    Serial.println("WiFi Access Point started.");
    Serial.print("ESP32 IP: ");
    Serial.println(WiFi.softAPIP());

    // Rest of the setup remains the same
    pinMode(motorEnablePin, OUTPUT);
    digitalWrite(motorEnablePin, HIGH);

    pinMode(motorpinLF1, OUTPUT);
    pinMode(motorpinLF2, OUTPUT);
    pinMode(motorpinLB1, OUTPUT);
    pinMode(motorpinLB2, OUTPUT);
    pinMode(motorpinRF1, OUTPUT);
    pinMode(motorpinRF2, OUTPUT);
    pinMode(motorpinRB1, OUTPUT);
    pinMode(motorpinRB2, OUTPUT);

    Serial.println("UDP Server Starting...");

    if (udp.listen(udpPort)) {
        Serial.println("UDP server started.");
        udp.onPacket([](AsyncUDPPacket packet) {
            char command = packet.data()[0];
            Serial.printf("Received command: %c\n", command);

            switch (command) {
                case 'f': Forward(); delay(100); break;
                case 'b': Backward(); delay(100); break;
                case 'l': Left(); delay(50); break;
                case 'r': Right(); delay(50); break;
                case 's': Stop(); break;
                default: Serial.println("Invalid Command");
            }

            Stop();

            udp.writeTo((const uint8_t*)"Command executed", 16, packet.remoteIP(), packet.remotePort());
        });
    }
}


void loop() {
    // Nothing needed in loop, everything runs in the UDP callback
}

void Forward() {
    Serial.println("Moving Forward");
    digitalWrite(motorpinLF1, HIGH);
    digitalWrite(motorpinLF2, LOW);
    digitalWrite(motorpinLB1, HIGH);
    digitalWrite(motorpinLB2, LOW);
    digitalWrite(motorpinRF1, HIGH);
    digitalWrite(motorpinRF2, LOW);
    digitalWrite(motorpinRB1, HIGH);
    digitalWrite(motorpinRB2, LOW);
}

void Backward() {
    Serial.println("Moving Backward");
    digitalWrite(motorpinLF1, LOW);
    digitalWrite(motorpinLF2, HIGH);
    digitalWrite(motorpinLB1, LOW);
    digitalWrite(motorpinLB2, HIGH);
    digitalWrite(motorpinRF1, LOW);
    digitalWrite(motorpinRF2, HIGH);
    digitalWrite(motorpinRB1, LOW);
    digitalWrite(motorpinRB2, HIGH);
}

void Left() {
    Serial.println("Turning Left");
    digitalWrite(motorpinLF1, LOW);
    digitalWrite(motorpinLF2, HIGH);
    digitalWrite(motorpinLB1, LOW);
    digitalWrite(motorpinLB2, HIGH);
    digitalWrite(motorpinRF1, HIGH);
    digitalWrite(motorpinRF2, LOW);
    digitalWrite(motorpinRB1, HIGH);
    digitalWrite(motorpinRB2, LOW);
}

void Right() {
    Serial.println("Turning Right");
    digitalWrite(motorpinLF1, HIGH);
    digitalWrite(motorpinLF2, LOW);
    digitalWrite(motorpinLB1, HIGH);
    digitalWrite(motorpinLB2, LOW);
    digitalWrite(motorpinRF1, LOW);
    digitalWrite(motorpinRF2, HIGH);
    digitalWrite(motorpinRB1, LOW);
    digitalWrite(motorpinRB2, HIGH);
}

void Stop() {
    Serial.println("Stopping");
    digitalWrite(motorpinLF1, LOW);
    digitalWrite(motorpinLF2, LOW);
    digitalWrite(motorpinLB1, LOW);
    digitalWrite(motorpinLB2, LOW);
    digitalWrite(motorpinRF1, LOW);
    digitalWrite(motorpinRF2, LOW);
    digitalWrite(motorpinRB1, LOW);
    digitalWrite(motorpinRB2, LOW);
}

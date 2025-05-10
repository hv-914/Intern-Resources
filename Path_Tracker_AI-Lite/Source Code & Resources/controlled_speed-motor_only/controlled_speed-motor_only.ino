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

// Motor pin definitions
#define motorpinLF1 14
#define motorpinLF2 16
#define motorpinLB1 32
#define motorpinLB2 33
#define motorpinRF1 19
#define motorpinRF2 17
#define motorpinRB1 25
#define motorpinRB2 26

// PWM properties
const int freq = 2400;      // PWM frequency
const int resolution = 8;   // 8-bit resolution (0-255)

// PWM channels for each motor pin
const int pwmChannelLF1 = 0;
const int pwmChannelLF2 = 1;
const int pwmChannelLB1 = 2;
const int pwmChannelLB2 = 3;
const int pwmChannelRF1 = 4;
const int pwmChannelRF2 = 5;
const int pwmChannelRB1 = 6;
const int pwmChannelRB2 = 7;

// Default motor speed (0-255) - Set to 200
int motorSpeed = 200;

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

    // Configure motor enable pin
    pinMode(motorEnablePin, OUTPUT);
    digitalWrite(motorEnablePin, HIGH);

    // Configure PWM for all motor pins
    ledcSetup(pwmChannelLF1, freq, resolution);
    ledcSetup(pwmChannelLF2, freq, resolution);
    ledcSetup(pwmChannelLB1, freq, resolution);
    ledcSetup(pwmChannelLB2, freq, resolution);
    ledcSetup(pwmChannelRF1, freq, resolution);
    ledcSetup(pwmChannelRF2, freq, resolution);
    ledcSetup(pwmChannelRB1, freq, resolution);
    ledcSetup(pwmChannelRB2, freq, resolution);
    
    // Attach PWM channels to GPIO pins
    ledcAttachPin(motorpinLF1, pwmChannelLF1);
    ledcAttachPin(motorpinLF2, pwmChannelLF2);
    ledcAttachPin(motorpinLB1, pwmChannelLB1);
    ledcAttachPin(motorpinLB2, pwmChannelLB2);
    ledcAttachPin(motorpinRF1, pwmChannelRF1);
    ledcAttachPin(motorpinRF2, pwmChannelRF2);
    ledcAttachPin(motorpinRB1, pwmChannelRB1);
    ledcAttachPin(motorpinRB2, pwmChannelRB2);

    Serial.println("UDP Server Starting...");

    if (udp.listen(udpPort)) {
        Serial.println("UDP server started.");
        udp.onPacket([](AsyncUDPPacket packet) {
            // Read the received data
            String packetData = "";
            for (int i = 0; i < packet.length(); i++) {
                packetData += (char)packet.data()[i];
            }
            
            // Improved command parsing to better support speed control
            // Format: <command>:<speed>
            // Examples: "f:150" (forward at speed 150), "b" (backward at default speed)
            int colonPos = packetData.indexOf(':');
            char command = packetData[0];
            
            // Check if speed setting is included
            if (colonPos > 0 && colonPos < packetData.length() - 1) {
                // Extract speed value after the colon
                String speedStr = packetData.substring(colonPos + 1);
                int newSpeed = speedStr.toInt();
                
                // Validate and set the speed
                if (newSpeed >= 0 && newSpeed <= 255) {
                    motorSpeed = newSpeed;
                    Serial.printf("Motor speed set to: %d\n", motorSpeed);
                }
            }

            Serial.printf("Received command: %c, Speed: %d\n", command, motorSpeed);

            switch (command) {
                case 'f': Forward(); break;
                case 'b': Backward(); break;
                case 'l': Left(); break;
                case 'r': Right(); break;
                case 's': Stop(); break;
                default: Serial.println("Invalid Command");
            }

            udp.writeTo((const uint8_t*)"Command executed", 16, packet.remoteIP(), packet.remotePort());
        });
    }
}

void loop() {
    // Nothing needed in loop, everything runs in the UDP callback
}

void Forward() {
    Serial.println("Moving Forward");
    ledcWrite(pwmChannelLF1, motorSpeed);
    ledcWrite(pwmChannelLF2, 0);
    ledcWrite(pwmChannelLB1, motorSpeed);
    ledcWrite(pwmChannelLB2, 0);
    ledcWrite(pwmChannelRF1, motorSpeed);
    ledcWrite(pwmChannelRF2, 0);
    ledcWrite(pwmChannelRB1, motorSpeed);
    ledcWrite(pwmChannelRB2, 0);
}

void Backward() {
    Serial.println("Moving Backward");
    ledcWrite(pwmChannelLF1, 0);
    ledcWrite(pwmChannelLF2, motorSpeed);
    ledcWrite(pwmChannelLB1, 0);
    ledcWrite(pwmChannelLB2, motorSpeed);
    ledcWrite(pwmChannelRF1, 0);
    ledcWrite(pwmChannelRF2, motorSpeed);
    ledcWrite(pwmChannelRB1, 0);
    ledcWrite(pwmChannelRB2, motorSpeed);
}

void Left() {
    Serial.println("Turning Left");
    ledcWrite(pwmChannelLF1, 0);
    ledcWrite(pwmChannelLF2, motorSpeed);
    ledcWrite(pwmChannelLB1, 0);
    ledcWrite(pwmChannelLB2, motorSpeed);
    ledcWrite(pwmChannelRF1, motorSpeed);
    ledcWrite(pwmChannelRF2, 0);
    ledcWrite(pwmChannelRB1, motorSpeed);
    ledcWrite(pwmChannelRB2, 0);
}

void Right() {
    Serial.println("Turning Right");
    ledcWrite(pwmChannelLF1, motorSpeed);
    ledcWrite(pwmChannelLF2, 0);
    ledcWrite(pwmChannelLB1, motorSpeed);
    ledcWrite(pwmChannelLB2, 0);
    ledcWrite(pwmChannelRF1, 0);
    ledcWrite(pwmChannelRF2, motorSpeed);
    ledcWrite(pwmChannelRB1, 0);
    ledcWrite(pwmChannelRB2, motorSpeed);
}

void Stop() {
    Serial.println("Stopping");
    ledcWrite(pwmChannelLF1, 0);
    ledcWrite(pwmChannelLF2, 0);
    ledcWrite(pwmChannelLB1, 0);
    ledcWrite(pwmChannelLB2, 0);
    ledcWrite(pwmChannelRF1, 0);
    ledcWrite(pwmChannelRF2, 0);
    ledcWrite(pwmChannelRB1, 0);
    ledcWrite(pwmChannelRB2, 0);
}
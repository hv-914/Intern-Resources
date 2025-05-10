#include <WiFi.h>
#include <ArduinoOTA.h>

// AP Credentials
const char *ssid = "106-69";     // Access Point SSID
const char *password = "12345678"; // AP Password (min 8 characters)
 
// Static IP configuration
IPAddress local_IP(192, 168, 106, 1);
IPAddress gateway(192, 168, 106, 1);
IPAddress subnet(255, 255, 255, 0);

void setup() {
    Serial.begin(115200);

    // Configure static IP for AP
    WiFi.softAPConfig(local_IP, gateway, subnet);

    // Start ESP32 as an Access Point
    WiFi.softAP(ssid, password);
    Serial.println("Access Point Started");
    Serial.print("IP Address: ");
    Serial.println(WiFi.softAPIP()); // Should print 192.168.106.69

    // Setup OTA
    ArduinoOTA.onStart([]() {
        Serial.println("OTA Update Started...");
    });

    ArduinoOTA.onEnd([]() {
        Serial.println("\nOTA Update Complete!");
    });

    ArduinoOTA.onProgress([](unsigned int progress, unsigned int total) {
        Serial.printf("Progress: %u%%\r", (progress * 100) / total);
    });

    ArduinoOTA.onError([](ota_error_t error) {
        Serial.printf("Error [%u]: ", error);
        if (error == OTA_AUTH_ERROR) Serial.println("Auth Failed");
        else if (error == OTA_BEGIN_ERROR) Serial.println("Begin Failed");
        else if (error == OTA_CONNECT_ERROR) Serial.println("Connect Failed");
        else if (error == OTA_RECEIVE_ERROR) Serial.println("Receive Failed");
        else if (error == OTA_END_ERROR) Serial.println("End Failed");
    });

    ArduinoOTA.begin();
    Serial.println("OTA Ready");
}

void loop() {
    ArduinoOTA.handle(); // Check for OTA updates
}

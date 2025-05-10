#include "HUSKYLENS.h"
#include <ESP32Servo.h>

int lastID = -1;
int servo_center = 100;//deg
int left_turn_angle = servo_center - 20; //80 deg
int right_turn_angle = servo_center + 20;//120 deg
HUSKYLENS huskylens;

// Servo Motor
#define SERVO_PIN 27
Servo servo;

// DC Motor
const int motorPin1 = 32; 
const int motorPin2 = 33; 
const int nslp = 13; 
const int frequency = 5000;

// DC Motor Functions
void motor_forward(int speed) {
  ledcWrite(5, speed);
  ledcWrite(6, 0);
  //Serial.println("motor_forward");
}

void motor_backward(int speed) {
  ledcWrite(5, 0);
  ledcWrite(6, speed);
  //Serial.println("motor_backward");
}

void motor_stop() {
  ledcWrite(5, 0);
  ledcWrite(6, 0);
  //Serial.println("motor_stop");
}
void moveServoTo(int angle) {
  // Constrain the angle between 0 and 180 degrees
  angle = constrain(angle, 75, 125);
  servo.write(angle);
  //delay(15);
  //Serial.println("Servo Angle : "+String(angle));
}

int husky() {
    if (huskylens.request() && huskylens.isLearned() && huskylens.available()) {
        HUSKYLENSResult result = huskylens.read();  // Read first result only
        return result.ID;
    }
    return -1;
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    while (!huskylens.begin(Wire)) {
        Serial.println(F("Begin failed!"));
        delay(100);
    }
    servo.attach(SERVO_PIN, 500, 2400);
      //######### DC Motor Setup ###########//
    ledcSetup(5, frequency, 8);
    ledcSetup(6, frequency, 8);
    ledcAttachPin(motorPin1, 5);
    ledcAttachPin(motorPin2, 6);
    pinMode(nslp, OUTPUT);
    digitalWrite(nslp, HIGH);
    //initial servo angle
    moveServoTo(servo_center);
}

void loop() {
    int currentID = husky();

    if (currentID != -1 && (currentID == 1 || currentID ==2)) {
        if (currentID == 1){
            Serial.print("Turning left: ");
            Serial.println(currentID);

            motor_backward(220);
            delay(500);
            motor_forward(220);
            moveServoTo(left_turn_angle);  // Left Turn
            delay(1000);
            moveServoTo(right_turn_angle); // Right Turn
            delay(500);
            moveServoTo(servo_center);

        } else if (currentID == 2){
            Serial.print("Turning right: ");
            Serial.println(currentID);

            motor_backward(220);
            delay(500);
            motor_forward(220);
            moveServoTo(right_turn_angle); // Right Turn
            delay(1000);
            moveServoTo(left_turn_angle);  // Left Turn
            delay(500);
            moveServoTo(servo_center);
        
        }
    }
    delay(1800);
    moveServoTo(servo_center);
    motor_stop();
}
#include <SoftwareSerial.h>
SoftwareSerial Bluetooth(3, 9); // RX, TX

//a1=ruote sinistra indietro
//a2=ruote sinistra avanti
//b1=ruote destra indietro
//b2=ruote destra avanti
const int motorA1  = 5;  // Pin  2 of L293
const int motorA2  = 6;  // Pin  7 of L293
const int motorB1  = 10; // Pin 10 of L293
const int motorB2  = 11;  // Pin 11 of L293
char command;   //dove salvo i comandi ricevuti

int s = 200;   // Default speed, from 0 to 255

void setup()
{
  pinMode(motorA1, OUTPUT);
  pinMode(motorA2, OUTPUT);
  pinMode(motorB1, OUTPUT);
  pinMode(motorB2, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);

  Serial.begin(9600);
  Bluetooth.begin(9600);
}

void loop() {
  
  if (Bluetooth.available())
  {
    
    String x = Bluetooth.readString();
    Bluetooth.println(x);
    Serial.println(x);
    digitalWrite(LED_BUILTIN, LOW);

    switch (x[0])    //valuto cosa fare   'a' spegne , 'b' accende
    {
      case '2'://sinistra sul posto
        Serial.println("Case a");
        analogWrite(motorA1, s);
        analogWrite(motorA2, 0);
        analogWrite(motorB1, 0);
        analogWrite(motorB2, s);
        break;

      case '0'://avanti
        Serial.println("Case w");
        analogWrite(motorA1, 0);
        analogWrite(motorA2, s);
        analogWrite(motorB1, 0);
        analogWrite(motorB2, s);
        break;

      case '1'://indietro
        Serial.println("Case s");
        analogWrite(motorA1, s);
        analogWrite(motorA2, 0);
        analogWrite(motorB1, s);
        analogWrite(motorB2, 0);
        break;

      case '3'://destra sul posto
        Serial.println("Case d");
        analogWrite(motorA1, 0);
        analogWrite(motorA2, s);
        analogWrite(motorB1, s);
        analogWrite(motorB2, 0);
        break;

      case '9'://stop
        Serial.println("Case spazio");
        analogWrite(motorA1, 0);
        analogWrite(motorA2, 0);
        analogWrite(motorB1, 0);
        analogWrite(motorB2, 0);
        break;
    }
  }
}

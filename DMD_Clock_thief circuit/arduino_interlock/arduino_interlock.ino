/* 
Interlock code for DMD Mirror Clock Pulse (MCP) Interrupter circuit
author: Daniel Harrington
8/2024

MCP blocked when MCP_control_pin pulls MOSFET HIGH

max_on_ms - defines the longest time the MCP should be interrupted
min_off_ms - minimum time the MCP interrupter should be off to allow sufficient time 
  for pulses to reach the DMD.

When TTL signal is HIGH, Arduino pulls the MOSFET HIGH. If signal has been on for longer
than max_on_ms, the MOSFET is pulled low for at least min_off_ms before being allowed to
be turned on again. This is to allow periodic pulses to the DMD to prevent mirror sticking.
If the TTL signal is kept HIGH, this will result in a switching of the MOSFET: on for max_on_ms,
off for min_off_ms. After the signal falls to LOW, the pulse cannot be interrupted until
min_off_ms time has passed.
*/

const int TTL_input_pin = 5;
const int MCP_control_pin = 4;
const long max_on_ms = 1000; // Maximum time the MCP can be interrupted, prevents mirror sticking
const long min_off_ms = 1000;

unsigned long prev_turnon_ms = 0;
unsigned long off_time_ms = 0;
int TTL_state;
int current_state = LOW;


void setup() {
  pinMode(TTL_input_pin, INPUT);
  pinMode(MCP_control_pin, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  // Initialize to off
  digitalWrite(MCP_control_pin, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(9600);
  Serial.println("hello");

}

void loop() {

  unsigned long current_ms = millis();
    
  // If been off for longer than the prescribed safety time, want to turn back on (set MOSFET low)
  bool time_exceeded = current_ms - prev_turnon_ms > max_on_ms;

  if (time_exceeded && current_state == HIGH) {
    // Pull MOSFET low and turn off LED, allow MCP to pass
    Serial.println("Time exceeded, turning off.");
    digitalWrite(MCP_control_pin, LOW);
    digitalWrite(LED_BUILTIN, LOW);
    off_time_ms = current_ms;
    current_state = LOW;
  }
  else {
    TTL_state = digitalRead(TTL_input_pin);

    if (TTL_state == LOW) {
      if (current_state == HIGH)
        off_time_ms = current_ms;
      Serial.println("Input LOW, turning off.");
      digitalWrite(MCP_control_pin, LOW);
      digitalWrite(LED_BUILTIN, LOW);
      current_state = LOW;
    }

    else if (TTL_state == HIGH && current_ms - off_time_ms > min_off_ms) {
      if (current_state == LOW)
        prev_turnon_ms = current_ms;
      Serial.println("Input HIGH, turning on.");
      digitalWrite(MCP_control_pin, HIGH);
      digitalWrite(LED_BUILTIN, HIGH);
      current_state = HIGH;
      if (current_state == HIGH)
        Serial.println("current_state HIGH");
    }
  }

}






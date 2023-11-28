# EloquentTensorFlow32

An Arduino library to run TensorFlow models on ESP32 chips without pain.

## How to use

Once you have your TensorFlow model exported in a C header format (for example using `xxd`),
running it is as easy as:

```cpp
#include <eloquent_tensorflow32.h>
#include "your_tf_model.h"
#define NUM_OPS 1
#define ARENA_SIZE 2000

using Eloquent::Esp32::TensorFlow;
TensorFlow<NUM_OPS, ARENA_SIZE> tf;

void setup() {
  Serial.begin(115200);

  tf.setNumInputs(1);
  tf.setNumOutputs(1);

  // init model
  while (!tf.begin(your_tf_model).isOk()) 
    Serial.println(tf.exception.toString());
}

void loop() {
  // fill your input vector
  float x[1] = {0};
  
  while (!tf.predict(x).isOk())
    Serial.println(tf.exception.toString());

  // one output
  Serial.print("One output: ");
  Serial.println(tf.result());

  // many outputs
  Serial.print("Many outputs: ");

  for (int i = 0; i < tf.numOutputs; i++) {
    Serial.print(tf.result(i));
    Serial.print(", ");
  }

  Serial.println();
}
```
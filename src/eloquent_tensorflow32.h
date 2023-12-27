#ifndef ELOQUENT_TENSORFLOW_32
#define ELOQUENT_TENSORFLOW_32

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "./exception.h"

using Eloquent::Extra::Exception;
using tflite::Model;
using tflite::ErrorReporter;
using tflite::MicroErrorReporter;
using tflite::MicroMutableOpResolver;
using tflite::MicroInterpreter;

namespace Eloquent {
    namespace Esp32 {
        /**
         * Run TensorFlow models the Eloquent-style
         */
        template<uint8_t numOps, uint16_t tensorArenaSize>
        class TensorFlow {
            public:
                const Model *model;
                ErrorReporter *reporter;
                MicroMutableOpResolver<numOps> resolver;
                MicroInterpreter *interpreter;
                TfLiteTensor *input;
                TfLiteTensor *output;
                Exception exception;
                uint8_t arena[tensorArenaSize];
                uint16_t numInputs;
                uint16_t numOutputs;
                float *outputs;

                /**
                 * Constructor
                 */
                TensorFlow() : 
                    exception("TF"),
                    reporter(nullptr),
                    model(nullptr),
                    interpreter(nullptr),
                    input(nullptr),
                    output(nullptr),
                    numInputs(0),
                    numOutputs(0),
                    outputs(NULL)
                {

                }

                /**
                 * 
                 */
                void setNumInputs(uint16_t n) {
                    numInputs = n;
                }

                /**
                 * 
                 */
                void setNumOutputs(uint16_t n) {
                    numOutputs = n;
                }

                /**
                 * Get i-th output
                 */
                float result(uint16_t i = 0) {
                    if (outputs == NULL || i >= numOutputs)
                        return sqrt(-1);

                    return outputs[i];
                }

                /**
                 * Init model
                 */
                Exception& begin(const unsigned char *data) {
                    if (!numInputs)
                        return exception.set("You must set the number of inputs");

                    if (!numOutputs)
                        return exception.set("You must set the number of outputs");

                    model = tflite::GetModel(data);

                    if (model->version() != TFLITE_SCHEMA_VERSION)
                        return exception.set(String("Model version mismatch. Expected ") + TFLITE_SCHEMA_VERSION + ", got " + model->version());

                    reporter = new MicroErrorReporter();
                    interpreter = new MicroInterpreter(model, resolver, arena, tensorArenaSize, reporter);

                    TfLiteStatus status = interpreter->AllocateTensors();

                    if (status != kTfLiteOk)
                        return exception.set("AllocateTensors() failed");

                    input = interpreter->input(0);
                    output = interpreter->output(0);

                    return exception.clear();
                }

                /**
                 * 
                 */
                template<typename T>
                Exception& predict(T *x) {
                    // quantize
                    float inputScale = input->params.scale;
                    float inputOffset = input->params.zero_point;

                    for (uint16_t i = 0; i < numInputs; i++)
                        input->data.int8[i] = (x[i] / inputScale) + inputOffset;

                    // execute
                    TfLiteStatus status = interpreter->Invoke();

                    if (status != kTfLiteOk)
                        return exception.set("Invoke() failed");

                    // allocate outputs
                    if (outputs == NULL)
                        outputs = (float*) calloc(numOutputs, sizeof(float));

                    // dequantize
                    float outputScale = output->params.scale;
                    float outputOffset = output->params.zero_point;

                    for (uint16_t i = 0; i < numOutputs; i++)
                        outputs[i] = (output->data.int8[0] - outputOffset) * outputScale;

                    return exception.clear();
                }

            protected:



        };
    }
}


#endif
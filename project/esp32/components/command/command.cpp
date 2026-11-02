#include "command.h"
#include "kws_config.h"
#include "kws_model_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "esp_heap_caps.h"

#include "esp_log.h"
#include <math.h>
#include <string.h>


static const char *TAG = "COMMAND";
static const tflite::Model *model = nullptr;
//128KB RAM for TensorFlow Lite Micro
static constexpr size_t TENSOR_ARENA_SIZE = 96*1024;
//Pointer point to RAM area of Tensor Arena
static uint8_t *tensor_arena = nullptr;
//Pointer point to TensorFlow Lite Micro
static tflite::MicroInterpreter *interpreter = nullptr;
//Resolver include operatorthat model can use
static tflite::MicroMutableOpResolver<9> resolver;
//Ensure command is called only one time
static bool initialized = false;

/**
 * @brief Initialize the TensorFlow Lite Micro keyword spotting model.
 *
 * Validates model schema, tensor dimensions, data types, and quantization
 * parameters before allowing inference. The tensor arena is allocated
 * from internal 8-bit-addressable RAM.
 *
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
extern "C" esp_err_t command_init(void) {
    if (initialized) {
        ESP_LOGI(TAG, "KWS already initialized");
        return ESP_OK;
    }

    ESP_LOGI(TAG, "===== KWS INITIALIZATION =====");
    ESP_LOGI(TAG, "Model size: %u bytes", (unsigned int)g_kws_model_len);
    ESP_LOGI(TAG, "Sample rate: %d Hz", KWS_SAMPLE_RATE);
    ESP_LOGI(TAG, "Window samples: %d", KWS_WINDOW_SAMPLES);
    ESP_LOGI(TAG, "Feature shape: %d x %d", KWS_N_FRAMES, KWS_N_MEL);
    ESP_LOGI(TAG, "Number of classes: %d", KWS_NUM_CLASSES);

    //Check model size
    if(g_kws_model_len < 8) {
        ESP_LOGE(TAG, "Model data is too small");
        return ESP_FAIL;
    }
    //Read model
    model = tflite::GetModel(g_kws_model);
    if(model == nullptr) {
        ESP_LOGE(TAG, "GetModel Failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "GetModel OK");

    //Check schema
    if(model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Schema mismatch: model = %u, runtime = %d", 
                    (unsigned int)model->version(),
                    TFLITE_SCHEMA_VERSION
                );
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Schema version: %u", (unsigned int)model->version());
    
    //Check if RAM still free
    ESP_LOGI(TAG, "Free internal RAM: %u bytes", (unsigned int)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    ESP_LOGI(TAG, "Largest free block: %u bytes", (unsigned int)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));

    // Register TFLite operators
    if (resolver.AddConv2D() != kTfLiteOk ||
        resolver.AddDepthwiseConv2D() != kTfLiteOk ||
        resolver.AddAveragePool2D() != kTfLiteOk ||
        resolver.AddFullyConnected() != kTfLiteOk ||
        resolver.AddReshape() != kTfLiteOk ||
        resolver.AddSoftmax() != kTfLiteOk ||
        resolver.AddAdd() != kTfLiteOk ||
        resolver.AddMul() != kTfLiteOk ||
        resolver.AddMean()!= kTfLiteOk) {

        ESP_LOGE(TAG, "Failed to register TFLite operators");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "TFLite operators registered");

    //Allocate Tensor Arena
    tensor_arena = (uint8_t *)heap_caps_malloc(
        TENSOR_ARENA_SIZE, 
        MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT
    );
    if(tensor_arena == nullptr) {
        ESP_LOGE(TAG, "Tensor Arena allocation failed");
        return ESP_ERR_NO_MEM;
    }
    ESP_LOGI(TAG, "Tensor Arena allocated: %u bytes", (unsigned int)TENSOR_ARENA_SIZE);
        

    //Create MicroInterpreter
    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );
    interpreter = &static_interpreter;
    ESP_LOGI(TAG, "MicroInterpreter created");

    //Allocate tensors
    TfLiteStatus status = interpreter->AllocateTensors();
    if(status != kTfLiteOk) {
        ESP_LOGE(TAG, "Allocate Tensors failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Allocate Tensors OK");
    ESP_LOGI(TAG, "Tensor Arena used: %u bytes", 
                (unsigned int)interpreter->arena_used_bytes()
            );

    // Get input and output tensors
    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);

    if (input == nullptr || output == nullptr) {
        ESP_LOGE(TAG, "Cannot get input/output tensors");
        return ESP_FAIL;
    }

    // Check INT8 data types
    if (input->type != kTfLiteInt8 ||
        output->type != kTfLiteInt8) {
        ESP_LOGE(TAG, "Model input/output is not INT8");
        return ESP_FAIL;
    }

    // Check input shape [1, 99, 40, 1]
    if (input->dims == nullptr ||
        input->dims->size != 4 ||
        input->dims->data[0] != 1 ||
        input->dims->data[1] != KWS_N_FRAMES ||
        input->dims->data[2] != KWS_N_MEL ||
        input->dims->data[3] != 1) {
        ESP_LOGE(TAG, "Incorrect input shape");
        return ESP_FAIL;
    }

    // Check output: 8 classes
    if (output->dims == nullptr ||
        output->dims->size != 2 ||
        output->dims->data[0] != 1 ||
        output->dims->data[1] != KWS_NUM_CLASSES) {
        ESP_LOGE(TAG, "Incorrect output shape");
        return ESP_FAIL;
    }
    if (output->bytes != KWS_NUM_CLASSES) {
        ESP_LOGE(TAG, "Incorrect output size");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Input shape: [1,%d,%d,1]",
                KWS_N_FRAMES, KWS_N_MEL);
    ESP_LOGI(TAG, "Output classes: %d",
                KWS_NUM_CLASSES);
    ESP_LOGI(TAG, "Input scale: %.9f",
                (double)input->params.scale);
    ESP_LOGI(TAG, "Input zero point: %d",
                (int)input->params.zero_point);
    ESP_LOGI(TAG, "Output scale: %.9f",
                (double)output->params.scale);
    ESP_LOGI(TAG, "Output zero point: %d",
                (int)output->params.zero_point);
    if(fabsf(input->params.scale - KWS_IN_SCALE) > 1e-6f || input->params.zero_point != KWS_IN_ZERO_POINT) {
        ESP_LOGE(TAG, "Input quantization mismatch");
        return ESP_FAIL;
    }
    if(fabsf(output->params.scale - KWS_OUT_SCALE) > 1e-6f || output->params.zero_point != KWS_OUT_ZERO_POINT) {
        ESP_LOGE(TAG, "Output quantization mismatch");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Quantization parameters OK");
    if (input->data.int8 == nullptr ||
        output->data.int8 == nullptr) {
        ESP_LOGE(TAG, "Tensor data pointer is null");
        return ESP_FAIL;
    }

    if (input->bytes != KWS_N_FRAMES * KWS_N_MEL) {
        ESP_LOGE(TAG, "Incorrect input tensor size");
        return ESP_FAIL;
    }


    initialized = true;
    ESP_LOGI(TAG, "Command component initialized");
    ESP_LOGI(TAG, "Free RAM after KWS init: %u bytes",
                (unsigned int)heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
            );
    ESP_LOGI(TAG, "===== KWS READY =====");
    return ESP_OK; 
}

/**
 * @brief Run keyword spotting inference on quantized Log-Mel features.
 *
 * Copies the input feature vector into the interpreter, invokes the
 * model, and returns the index of the largest INT8 output score.
 * A normalized score is also calculated from the output values.
 *
 * @param features Pointer to the INT8 Log-Mel feature array.
 * @param features_count Number of input features.
 * @param predicted_class Output pointer for the predicted class index.
 * @param confidence Output pointer for the calculated confidence score.
 * @return ESP_OK on success, or an ESP-IDF error code on failure.
 */
extern "C" esp_err_t command_predict(
    const int8_t *features,
    size_t features_count,
    int *predicted_class,
    float *confidence
) {
    //Check initialization
    if(!initialized || interpreter == nullptr) {
        ESP_LOGE(TAG, "KWS model not initialized");
        return ESP_ERR_INVALID_STATE;
    }
    
    //Check argument
    if(features == nullptr ||
       predicted_class == nullptr ||
       confidence == nullptr) {
            ESP_LOGE(TAG, "Invalid argument");
            return ESP_ERR_INVALID_ARG;
       }
    if(features_count != KWS_N_FRAMES*KWS_N_MEL) {
        ESP_LOGE(TAG, "Invalid feature count: %u", (unsigned int)features_count);
        return ESP_ERR_INVALID_SIZE;
    }

    //Get Tensor
    TfLiteTensor *input = interpreter->input(0);
    TfLiteTensor *output = interpreter->output(0);

    if(input == nullptr || 
       output == nullptr ||
       input->data.int8 == nullptr ||
       output->data.int8 == nullptr) {
            ESP_LOGE(TAG, "Invalid Tensor");
            return ESP_FAIL;
       }
    
    //Check input, output Tensor
    if (input->bytes != features_count) {
        ESP_LOGE(TAG, "Input tensor size mismatch");
        return ESP_ERR_INVALID_SIZE;
    }
    if (output->bytes != KWS_NUM_CLASSES) {
        ESP_LOGE(TAG, "Output tensor size mismatch");
        return ESP_ERR_INVALID_SIZE;
    }

    //Copy quantized Log-Mel features
    memcpy(input->data.int8, features, features_count);

    //Run inference
    TfLiteStatus status = interpreter->Invoke();
    if(status != kTfLiteOk) {
        ESP_LOGE(TAG, "Inference failed");
        return ESP_FAIL;
    }

    //Find predicted class using INT8 values
    int best_index = 0;
    int8_t best_q = output->data.int8[0];
    for(int i = 1; i < KWS_NUM_CLASSES; i++) {
        if(output->data.int8[i] > best_q) {
            best_q = output->data.int8[i];
            best_index = i;
        }
    }

    //Convert output logits to probabilities using Softmax
    float max_logit = (static_cast<int>(best_q) - output->params.zero_point) *output->params.scale;
    float sum_exp = 0.0f;
    for(int i = 0; i < KWS_NUM_CLASSES; i++) {
        float logit =(static_cast<int>(output->data.int8[i]) - output->params.zero_point)*output->params.scale;
        sum_exp += expf(logit - max_logit);
    }
    float best_probability = 1.0f / sum_exp;

    //Return result
    *predicted_class = best_index;
    *confidence = best_probability;
    
    return ESP_OK;
}




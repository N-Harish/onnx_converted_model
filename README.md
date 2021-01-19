# onnx_converted_model

* This is a demo model of mnist dataset which was trained using tensorflow 2.2.0 and then converted to onnx for performance
* The model in onnx format is light weight and gives faster inference
* The onnx model is 3x lighter than keras model and the infeence time is much faster as compared to keras
* The accuracy of both model are 99%

# Requirements

* keras2onnx
* onnx
* onnxruntime
* tensorflow==2.2.0

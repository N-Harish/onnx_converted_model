import onnx
from keras.models import load_model
import keras2onnx

k_model = keras.models.load_model('mnist.h5')
onnx_model = keras2onnx.convert_keras(k_model)
onnx.save_model(onnx_model, 'onnx_mnist.onnx')
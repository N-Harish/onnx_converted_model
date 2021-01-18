import onnx
import onnxruntime
import json
import matplotlib.pyplot as plt
from tensorflow import keras
import random

# load data from keras
(x_train1, y_train1), (x_test1, y_test1) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train1.astype("float32") / 255
x_test = x_test1.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)



data = json.dumps({'data': x_test.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession('onnx_mnist.onnx', None)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)

num = random.randrange(9999)
data1 = np.expand_dims(x_test[num],axis=0)

result1 = session.run([output_name], {input_name: data1})
prediction1 = int(np.argmax(np.array(result1).squeeze(), axis=0))
print(prediction1)

plt.imshow(x_test1[num])
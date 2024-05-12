import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('./bert.keras')
model.summary()
predictions = model.predict(np.array(["Meanwhile Trump won't even release his SAT scores and his Wharton professors said he was the dumbest student they've ever taught"]))

print(predictions)

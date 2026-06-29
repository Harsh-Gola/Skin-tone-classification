import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model =load_model('skin_tone_model.h5')

class_labels=[
    "black",
    'brown',
    'white'
]

def predict_skin_tone(img_path):
     img = image.load_img(
        img_path,
        target_size=(128,128)
    )
     
     img_array=image.img_to_array(img)
     img_array= img_array/255.0
     img_array=np.expand_dims(img_array,axis=0)
     

     prediction =model.predict(img_array)
     predicted_class=np.argmax(prediction)
     return class_labels[predicted_class]
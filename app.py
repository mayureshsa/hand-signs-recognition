from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import io
import json
app = Flask(__name__, template_folder='templates')

def init():
   global model
   model = load_model('model//History_HSRS_GlobalSignsDataset_19June_CNN12.h5')
   

@app.route('/')
def upload_file():
   return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
   if request.method == 'POST':
      img = request.files["file"].read()
      img = Image.open(io.BytesIO(img))
      image = img.resize((224, 224))
      image = img_to_array(image)
      image = image/255
      image = np.expand_dims(image, axis = 0)



      y_pred = model.predict_classes(image)
      
      return 'Predicted Number: ' + str(y_pred[0])
  
if __name__ == '__main__':
   print(("* Loading Keras model and Flask starting server..."
      "please wait until server has fully started"))
   init()
   app.run(debug = True)
   
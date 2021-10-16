from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import numpy as np
import tensorflow
from PIL import Image, ImageChops
import os 
import cv2
import glob



def ELA(img_path):

    """Performs Error Level Analysis over a directory of images"""
    
    TEMP = 'ela_' + 'temp.jpg'
    SCALE = 10
    original = Image.open(img_path)
    try:
        original.save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original, temporary)
    except:
        original.convert('RGB').save(TEMP, quality=90)
        temporary = Image.open(TEMP)
        diff = ImageChops.difference(original.convert('RGB'), temporary)
        
    d = diff.load()
    WIDTH, HEIGHT = diff.size
    for x in range(WIDTH):
        for y in range(HEIGHT):
            d[x, y] = tuple(k * SCALE for k in d[x, y])

    return diff

# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def detectFaces(path,f,dest):
  
    filenames = glob.glob(path+'/*.mp4')
    k=0

    for filename in filenames:
  
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    nof = v_len//f
  
    j=0
    cnt = 0
    while(cnt<f and j<v_len):

        v_cap.set(1,j)
        # Load frame
        success, img = v_cap.read()
        if not success:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        faces = faces[:1]
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Display
    
        cv2.imwrite(dest +'/'+ str(k)+'.jpg', img[y:y+h,x:x+w])
        k+=1
        j+=nof
        cnt+=1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

  
    v_cap.release()
    cv2.destroyAllWindows()

def prepare_image(file):

    img=image.load_img(file,target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array_expanded_dims=np.expand_dims(img_array,axis=0)
    return tensorflow.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

app = Flask(__name__)
uploaded_file = ''

app.config['image_upload'] = 'D:/ankush/projects/itsfake/uploads'
app.config['video_upload'] = 'D:/ankush/projects/itsfake/video_upload'
app.config['face'] = 'D:/ankush/projects/itsfake/face'

@app.route('/main',methods = ['GET','POST'])
def main():
    return render_template('main.html')


@app.route('/upload_file' , methods = ['POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files['data']
        data = request.form['fav_language']
        if (data=="image_"):
            uploaded_file.save(os.path.join(app.config['image_upload'],uploaded_file.filename))
            model = load_model('new_model_casia.h5')
            numpydata=np.array(ELA(uploaded_file).resize((128, 128))).flatten() / 255.0
            numpydata = np.resize(numpydata,(1,128,128,3))
            pred = model.predict(numpydata)     
            return render_template('./after.html',data = pred[0][0])
        elif (data== "deepfake_"):
            print("Deepfake")
            model2 = load_model('deep_fakes.h5')
            uploaded_file.save(os.path.join(app.config['video_upload'],uploaded_file.filename))
            detectFaces(app.config['video_upload'],20,app.config['face'])
            
            count=0
            files = os.listdir(app.config['face'])
            for file in files:
                preprocessed_image=prepare_image(app.config['face']+'/'+file)
                pred = model2.predict(preprocessed_image)
                print(pred)
                if pred[0][0] >= 0.5:
                    count+=1
                if count>10:
                    return render_template('./after.html',data = 0.1)    
                else:
                    return render_template('./after.html',data = 0.9)
        
                '''elif (data=='profile_'):
                    print("Profile")
                else:
                    print("else")
                    return render_template('after.html','something')'''
        else:
                return render_template('./after.html',data=0.0)

if __name__ == '__main__':
    app.run(debug=True)
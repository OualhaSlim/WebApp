from keras.models import model_from_json
from keras import backend as K
from flask import Flask, render_template, flash, redirect,url_for
from flask import request
from flask import jsonify
#import librosa
import pandas as pd
import numpy as np
from python_speech_features import mfcc
import os
from pydub import AudioSegment

app = Flask(__name__)


def get_model(path):
    K.clear_session()
    dir=os.getcwd()
    print(dir)
    json_file = open(os.path.join(dir,path+".json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(os.path.join(dir,path+".h5"))
    print('model loaded!')
    return model

def envelope(x, rate, threshold):
    mask = []
    x = pd.Series(x).apply(np.abs)
    y_mean = x.rolling(window=int(rate/10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def data_processing(signal):
    rate = 44100
    length = 4
    mask = envelope(signal,rate,0.001)
    aux = signal[mask] #remove amplitudes less than 0.001
    audio = []
    if(aux.shape[0]/rate <= length):#audio file less than 4 seconds
        new=np.zeros(length*rate)
        shape=2*aux.shape[0]
        new[:aux.shape[0]]=aux
        while((shape/rate)<length):
            new[shape-aux.shape[0]:shape]=aux
            shape+=aux.shape[0]
        shape-=aux.shape[0]
        diff=new[shape:].shape[0]
        new[shape:]=aux[:diff]
        audio.append(new)
        return audio,1
    else :
        shape=aux.shape[0]
        step=0
        counter = 0
        for i in range(0,int(shape/(length*rate))-1):
            new = np.zeros(length*rate)
            new[:]=aux[step:step+length*rate]
            audio.append(new)#wavfile.write(path+'/'+str(df.at[f,'counter'])+'_'+f, rate, new)
            counter +=1
            step+=length*rate
        if(shape%(length*rate)!=0):
            new=np.zeros(length*rate)
            new[:]=aux[shape-(length*rate):] #add the rest
            audio.append(new)#wavfile.write(path+'/'+str(df.at[f,'counter'])+'_'+f, rate, new)
            counter +=1
    print('signal preprocessed')
    return audio,counter

class Config:
    def __init__(self, winlen=0.023 , winstep=0.01 ,nfeat=13,nfilt=26, nfft=1024, rate=44100):
        self.winlen=winlen
        self.winstep=winstep
        self.nfeat = nfeat
        self.nfilt=nfilt
        self.nfft = nfft
        self.rate = rate

def build_mfcc_instrument(audio,counter,_min=-152.144687976,_max=180.49846957):
    c = Config()
    X = []
    if(counter==1):
        X_mfcc = mfcc(audio[0], 44100,
                      winlen=c.winlen, winstep=c.winstep, numcep=c.nfeat, nfilt=c.nfilt, nfft=c.nfft,
                      winfunc=lambda x: np.hamming(x)).T

        X.append(X_mfcc)

    else:
        for i in range(counter):
            X_mfcc = mfcc(audio[i], 44100,
                winlen=c.winlen,winstep=c.winstep,numcep=c.nfeat, nfilt=c.nfilt, nfft=c.nfft,
                winfunc=lambda x: np.hamming(x)).T
            X.append(X_mfcc)

    X= np.array(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    #print('feature ready')
    return X

def build_mfcc_env(audio,counter,_min=-134.038533911,_max= 146.7676106):
    c = Config()
    X = []
    if(counter==1):
        X_mfcc = mfcc(audio[0], 44100,
                      winlen=c.winlen, winstep=c.winstep, numcep=c.nfeat, nfilt=c.nfilt, nfft=c.nfft,
                      winfunc=lambda x: np.hamming(x)).T

        X.append(X_mfcc)
    else:
        for i in range(counter):
            X_mfcc = mfcc(audio[i], 44100,
                winlen=c.winlen,winstep=c.winstep,numcep=c.nfeat, nfilt=c.nfilt, nfft=c.nfft,
                winfunc=lambda x: np.hamming(x)).T
            X.append(X_mfcc)

    X= np.array(X)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num

def prediction(model,signal,list_unique,category):
    audio , counter = data_processing(signal)
    if(category):#env
        X = build_mfcc_env(audio=audio, counter=counter)
    else:#instruments
        X = build_mfcc_instrument(audio=audio,counter=counter)
    print('predicting...')
    All_pred = model.predict(X)
    pred_id = []
    for i in range(len(All_pred)):
        pred_id.append(All_pred[i].argmax())
    print('prediction done!')
    return list_unique[most_frequent(pred_id)]



#@app.route('/')
#def home():
 #   return render_template('index.html')


@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        message = ''
        f = request.files['file']
        if (request.form.get('classifier')=="instrument"):
            category = 0
        else:
            category = 1

        f.save(f.filename)

        list_instruments = ['Hi-hat', 'Saxophone', 'Trumpet', 'Glockenspiel', 'Cello', 'Clarinet', 'Snare_drum', 'Oboe',
                            'Flute'
            , 'Bass_drum', 'Harmonica', 'Gong', 'Double_bass', 'Tambourine', 'Electric_piano', 'Acoustic_guitar',
                            'Violin_or_fiddle']

        list_env = ['Knock', 'Gunshot_or_gunfire', 'Computer_keyboard',
                    'Keys_jangling', 'Writing', 'Laughter', 'Tearing',
                    'Fart', 'Cough', 'Telephone', 'Bark', 'Bus', 'Squeak', 'Scissors',
                    'Microwave_oven', 'Burping_or_eructation', 'Shatter',
                    'Fireworks', 'Cowbell', 'Meow', 'Chime',
                    'Drawer_open_or_close', 'Applause', 'Finger_snapping']
        name, ext = os.path.splitext(f.filename)
        if(ext == '.wav'):
            signal, rate = librosa.load(str(f.filename), sr=44100)

        else:
            sound = AudioSegment.from_mp3(f.filename)
            sound.export("file.wav", format="wav")
            signal, rate = librosa.load('file.wav', sr=44100)

        if (category):
            model = get_model("models\EnvModel")
            pred = prediction(model=model, signal=signal, category=category, list_unique=list_env)
        else:
            model = get_model('models\InstrumentsModel')
            pred = prediction(model=model, signal=signal, category=category, list_unique=list_instruments)
        os.remove(f.filename)
    else:
        return render_template('index.html')
    print("the prediction is")
    print(pred)
    #return pred
    flash('The prediction is'+str(pred))
    message = 'The prediction is'+str(pred)
    return render_template('index.html', message=message)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port)


# coding: utf-8

# In[1]:

import sounddevice as sd
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from pydub import AudioSegment
from pydub.silence import split_on_silence

#import time
import numpy as np
import glob
import os
from sys import argv
import numpy as np
import PIL.Image as Image
import threading
from array import array
from Queue import Queue, Full
import os
import scipy.fftpack
import numpy
import scipy
import time
import pyaudio
import wave
import contextlib
import sys
import keras
import cv2
from keras.models import load_model, Model


# In[2]:

keras.__version__


# In[2]:

TRAINED_MODEL_PATH ="path to trained model/LID-FRNN-DR-0.2-epoch- 7--0.92--0.21--val_acc-0.75-val_loss-0.81.h5"


# In[3]:

model = load_model(TRAINED_MODEL_PATH)


# In[4]:

def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
	win = window(frameSize)
	hopSize = int(frameSize - np.floor(overlapFac * frameSize))

	# zeros at beginning (thus center of 1st window should be for sample nr. 0)
	samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
	# cols for windowing
	cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize))) + 1
	# zeros at end (thus samples can be fully covered by frames)
	samples = np.append(samples, np.zeros(frameSize))

	frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
	frames *= win

	return np.fft.rfft(frames)    


# In[6]:

def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
	spec = spec[:, 0:256]
	timebins, freqbins = np.shape(spec)
	scale = np.linspace(0, 1, freqbins) #** factor

	scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale))
	scale *= (freqbins-1)/max(scale)

	newspec = np.complex128(np.zeros([timebins, freqbins]))
	allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
	freqs = [0.0 for i in range(freqbins)]
	totw = [0.0 for i in range(freqbins)]
	for i in range(0, freqbins):
		if (i < 1 or i + 1 >= freqbins):
			newspec[:, i] += spec[:, i]
			freqs[i] += allfreqs[i]
			totw[i] += 1.0
			continue
		else:
	       
			w_up = scale[i] - np.floor(scale[i])
			w_down = 1 - w_up
			j = int(np.floor(scale[i]))

			newspec[:, j] += w_down * spec[:, i]
			freqs[j] += w_down * allfreqs[i]
			totw[j] += w_down

			newspec[:, j + 1] += w_up * spec[:, i]
			freqs[j + 1] += w_up * allfreqs[i]
			totw[j + 1] += w_up

	for i in range(len(freqs)):
		if (totw[i] > 1e-6):
			freqs[i] /= totw[i]

		return newspec, freqs


# In[7]:

def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="gray", channel=0, name='tmp.png', alpha=1, offset=0):
	samplerate, samples = wav.read(audiopath)
	#samples = samples[:, channel]
	s = stft(samples, binsize)

	sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
	sshow = sshow[2:, :]
	ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
	timebins, freqbins = np.shape(ims)

	ims = np.transpose(ims)
	# ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
	ims = ims[0:256, :] # 0-11khz, ~10s interval
	#print "ims.shape", ims.shape

	image = Image.fromarray(ims) 
	image = image.convert('L')
	image.save(name)


CHUNK_SIZE = 1024
MIN_VOLUME = 5000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 12
BUF_MAX_SIZE = CHUNK_SIZE * 10


IMAGE_HEIGHT = 256
IMAGE_WIDTH = 860
IMAGE_CHANNELS = 1

num_classes = 3
predicted_classes = list()
predicted_probs = list()
WAVE_OUTPUT_FILENAME = "file.wav"
audio = pyaudio.PyAudio()
name=time.time()
   
        # start Recording
        
try:
    while(1) :


        print("please speak for 12 seconds without pauses")
        # start Recording
        print ("recording...")
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            #print data
        print ("finished recording")

        start = time.time() #Start Timing

        stream.stop_stream()
        stream.close()
        audio.terminate()
        a = str(time.time())

        waveFile = wave.open("/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1.wav", 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()


        # In[9]:

        #sound_file = AudioSegment.from_wav("/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1.wav")
        #audio_chunks = split_on_silence(sound_file, min_silence_len=500, silence_thresh = sound_file.dBFS-3)
        #if audio_chunks != None:
            #clip = audio_chunks[0][0:110]
            #for i, chunk in enumerate(audio_chunks):
                #clip = clip.append(chunk)
            #clip.set_frame_rate(44100)
            #clip.export("/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1_Scilence.wav", format="wav")


        # In[10]:

        clip = AudioSegment.from_wav("/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1.wav")
        clip = clip.set_frame_rate(44100)
        for i in range(0, int(clip.duration_seconds*1000-10000), 10000):
            chunks = clip[i:i+10000]
            chunks.export("/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1_Chunked.wav", format="wav")  


        # In[11]:


        wavfile = "/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1_Chunked.wav" 
                   
        plotstft(wavfile, name="/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1_Spectrogram.png", alpha=1.0)
                   

        # In[12]:

        file_path = "/home/telxsi.com/snigdha.n/Desktop/for_release/recorded_files/1_Spectrogram.png"


        # In[13]:

        img = cv2.imread(file_path, 0)
        imgs = img


        imgs = imgs.reshape(1,IMAGE_WIDTH, IMAGE_HEIGHT)
            


        # In[14]:

        classes = ['english','japanese','mandarin'] 
        num_classes = len(classes)
        #print "Total output classes: ", num_classes

        per_class_samples = [0] * num_classes


        # In[15]:

        print (classes)
        probabilities=model.predict(imgs)
        stop = time.time() #Stop Timing
        sorted_prob_idxs = (-probabilities).argsort()[0]
        predicted_prob = np.amax(probabilities)
        predicted_probs.append(predicted_prob)
        predicted_class = classes[sorted_prob_idxs[0]]
        predicted_classes.append(predicted_class)
        # In[16]:





        print (probabilities)


        # In[ ]:

        print(predicted_class)
        print "Time Taken : ",stop - start
        print "Do you want to speak again ? Press y or n :"
        status = raw_input()
        if (status =="n"):
            print "Program Exited"
            quit()


except KeyboardInterrupt:
    quit()     # In[ ]:



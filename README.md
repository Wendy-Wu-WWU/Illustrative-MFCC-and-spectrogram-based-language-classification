# Language-classification
Classification of various languages based on deep learning approach
Various language Data is downloaded from youtube. Mainly news channels. Audio without background score is prefered. If certain audio has background score, it can be removed in Audacity

Data is further preprocessed. Channel set to mono. sampling rate preffered = 44100 (can be changed) silence clipped. Each audio sample is taken as 10 sec (can be changed in data preprocesssing code)

Feature (spectrogram) can be extracted using one of the feature extractor code.
Spectrogram files should be named as follows :
Class 0 :
0_name1.jpg , 0_name2.jpg , 0_name3.jpg
Class1 :
1_name1.jpg , 1_name2.jpg , 1_name3.jpg
and so on
there the 3 different architecture in training code. the sutaible architecture can be selected
Model can be tested in live environment using testor code. The feature extractor and preprocessing should be same while training and testing.. Should be changed accordingly.

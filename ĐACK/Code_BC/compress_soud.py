import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import pickle
from scipy.io import wavfile
join = os.path.join


samplerate, data_full = wavfile.read('input.wav')
n = 4
w = 'haar'
keep = 0.1
for keep  in (0.1, 0.05, 0.01, 0.005):
    file4 = []
    for i in (0,1):
        coeffs_img = pywt.wavedec(data_full[:,[i]].reshape(1,-1),wavelet=w,level=n)
        
        Csort =[] 
        for i in range(n+1):
            Csort.append(np.sort(np.abs(coeffs_img[i].reshape(-1))))
        
        coeffs_img1 = []
        u = []
        thresh = []
        for i in range(n+1):
            thresh.append(Csort[i][int(np.floor((1-keep)*len(Csort[i])))])
        for i in range(n+1):    
            ind = np.abs(coeffs_img[i]) > thresh[i]
            coeffs_img1.append(coeffs_img[i] * ind)
            u.append(np.where(ind == True)) 
        
        data = []
        for i in range(n+1):
            I = []
            for r,c in zip(u[i][0],u[i][1]):
                I.append(coeffs_img1[i][r][c])
            data.append([u[i],I])
           
        Arecon = pywt.waverec(coeffs_img1,wavelet=w)
        file4.append(Arecon)
    wavfile.write(join('compress_so/wav','compres'+str(keep)+'.wav'),samplerate,np.array(file4).T.astype(np.int16))
    file = open(join('compress_so/pkl',str(keep)+'.pkl'),'wb')
    pickle.dump(data, file)
    file.close()
lisst = os.listdir
base_size=os.stat('input.wav').st_size
for comr in lisst('compress_so/pkl'):
    compress_size=os.stat(join('compress_so/pkl',comr)).st_size
    print('K= ',comr[:-4],'R = ', compress_size/base_size)
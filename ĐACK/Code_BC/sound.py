import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import pickle
from scipy.io import wavfile
join = os.path.join

samplerate, data_full = wavfile.read('input.wav')
B=data_full
w_l = pywt.families()
n = 4
w = 'haar'
w_l = pywt.wavelist(kind='discrete')
plt.figure()
plt.plot(B[:10000,[0]])
plt.xlabel('sample index')
plt.ylabel('amplitude')
for w in w_l:
    coeffs_img = pywt.wavedec(B[:10000,[0]].reshape(1,-1),wavelet=w,level=n)
    
    Csort =[] 
    for i in range(n+1):
        Csort.append(np.sort(np.abs(coeffs_img[i].reshape(-1))))
    
    
    for keep in ( [0.1]):#, 0.05, 0.01, 0.005):
        coeffs_img1 = []
        u = []
        thresh =[]
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
         
        file = open(join('sound_pkl',str(keep)+'.pkl'),'wb')
        pickle.dump(data, file)
        file.close()
          
    
        fig1, axs1 = plt.subplots(1,1)
        a = coeffs_img1[0]
        axs1.set_title('cA1')
        axs1.plot(a.reshape(-1,1))
    
        
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                                subplot_kw={'xticks': [], 'yticks': []})
        
        axs = axs.flat
        inde = [0,2]
        for index in inde:
            if index == 0:
                axs[index].set_title('cD'+ str(index+1))
                axs[index + 1].set_title('cD'+ str(index+2))
            else:
                axs[index].set_title('cD'+ str(index+1))
                axs[index + 1].set_title('cD'+ str(index+2))
            axs[index].plot(coeffs_img1[index+1].reshape(-1,1))
            axs[index + 1].plot(coeffs_img1[index+2].reshape(-1,1))
    
        
         
        Arecon = pywt.waverec(coeffs_img1,wavelet=w)
        plt.figure()
        plt.title('reconstruct_'+w)
        plt.plot(Arecon.reshape(-1,1))

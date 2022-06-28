import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import pickle
join = os.path.join
plt.rcParams['figure.figsize'] = [8, 8]
plt.rcParams.update({'font.size': 18})

A = cv2.imread('dog.bmp')
B = np.mean(A, -1)
w_l = pywt.wavelist(kind='discrete')
n = 4
w = 'haar'
for w in w_l:
    coeffs_img = pywt.wavedec2(B,wavelet=w,level=n)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs_img)
    Csort = np.sort(np.abs(coeff_arr.reshape(-1)))
    # for keep in (0.1, 0.05, 0.01, 0.005):
    for keep in ([0.005]):
        thresh = Csort[int(np.floor((1-keep)*len(Csort)))]
        ind = np.abs(coeff_arr) > thresh
        Cfilt = coeff_arr * ind
        u = np.where(ind == True)
        coeffs_filt = pywt.array_to_coeffs(Cfilt,coeff_slices,output_format='wavedec2')
        data = []
        for i in range(n+1):
            
            I = []
            if i == 0:
                u = np.where(np.abs(coeffs_filt[i]) > 1)
                for r,c in zip(u[0],u[1]):
                    I.append(coeffs_filt[i][r][c].astype('uint8'))
                data.append([u,I])   
            else:
                for l in range(3):
                    I1 = []
                    u1 = np.where(np.abs(coeffs_filt[i][l]) > 1)
                    for r,c in zip(u1[0],u1[1]):
                        try:
                            I1.append(coeffs_filt[i][l][r][c].astype('uint8'))
                        except:
                            print(r,c,i,l)
                    data.append([u1,I1]) 
          
        file = open(join('data_pkl',str(keep)+'.pkl'),'wb')
        pickle.dump(data, file)
        file.close()
                
        fig1, axs1 = plt.subplots(1,1)
        a = coeffs_filt[0]
        axs1.imshow(a.astype('uint8'),cmap='gray')
        
        fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(10, 10),
                                subplot_kw={'xticks': [], 'yticks': []})
        
        axs = axs.flat
        inde = [0,3,6,9]
        for i, index in zip(range(n),inde):
            v = coeffs_filt[i+1]
            axs[index].imshow(v[0].astype('uint8'),cmap='gray')  
            axs[index + 1].imshow(v[1].astype('uint8'),cmap='gray')
            axs[index + 2].imshow(v[2].astype('uint8'),cmap='gray')
        
         
        Arecon = pywt.waverec2(coeffs_filt,wavelet=w)
        plt.figure()
        plt.title(w)
        plt.imshow(Arecon.astype('uint8'),cmap='gray')
        cv2.imwrite(join('img_comprees',str(keep)+'.bmp'),Arecon.astype('uint8'))
lisst = os.listdir
base_size=os.stat('pokemon.bmp').st_size
for comr in lisst('compress_so/pkl'):
    compress_size=os.stat(join('compress_so/pkl',comr)).st_size
    print('K= ',comr[:-4],'R = ', compress_size/base_size)
print('----------------------------')
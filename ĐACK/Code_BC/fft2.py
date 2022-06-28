from matplotlib.image import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import cv2
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams.update({'font.size': 18})
join = os.path.join

B=cv2.imread("dog.bmp",0)

Bt = np.fft.fft2(B)
Btsort = np.sort(np.abs(Bt.reshape(-1))) # sort by magnitude

# Zero out all small coefficients and inverse transform
for keep in (0.1, 0.05, 0.02, 0.005):
    thresh = Btsort[int(np.floor((1-keep)*len(Btsort)))]
    ind = np.abs(Bt)>thresh          # Find small indices
    Atlow = Bt * ind                 # Threshold small indices
    u1 = np.where(ind == True)
    print(u1)
    I = []
    data = []
    for l in range(3):
                I1 = []
                for r,c in zip(u1[0],u1[1]):
                    try:
                        I1.append(Atlow[l][r][c])
                    except:
                        print(r,c,l)
                data.append([u1,I1]) 
    file = open('k'+str(keep)+'fft'+'.pkl','wb')
    pickle.dump(data, file)
    file.close()
    Alow = np.fft.ifft2(Atlow).real  # Compressed image
    
    plt.figure()
    plt.imshow(Alow,cmap='gray')
    plt.axis('off')
    plt.title('Compressed image: keep = ' + str(keep))
    cv2.imwrite(str(keep)[-3:]+'ttf'+'.bmp',Alow.astype('uint8'))
#---------Tính tỉ số nén R---------
#B1: Lấy kích thước của file
import os
base_size=os.stat('pokemon.bmp').st_size
compress_size=os.stat('k0.02fft.pkl').st_size

print(base_size)
print(compress_size)

#B2: Tính tỉ số nén, in ra tỉ số nén
print('K= ',0.02,'R = ', compress_size/base_size)
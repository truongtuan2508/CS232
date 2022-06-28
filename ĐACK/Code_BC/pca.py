import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

#PCA
def PCA(X,K):
    #B1: Tìm điểm nằm ở giữa:
    X_mean=X.mean(axis=0)
    #B2 Dời tập các điểm về gốc tọa độ:
    X_hat=X-X_mean
    #B3: Tìm phương sai
    S=np.dot(X_hat.T,X_hat)/len(X_hat.T[0])
    #B4: Tìm trị riêng
    lamb,U=np.linalg.eig(S)
    ind=np.argsort(lamb[::-1])
    U=U[:,ind]
    #B5: Chọn k vector riêng
    U_k=U[:,:K]
    #B6: Chiếu điểm lên các vector riêng
    Z=np.dot(U_k.T,X_hat.T)
    #B7: Điểm được chiếu cũng chính là kết quả
    return U_k,Z,X_mean

def decode(U_k,Z,X_mean):
    X_star=np.dot(U_k,Z)+X_mean.reshape(-1,1)    
    return X_star


percent=0.02
#init
img=cv2.imread("pokemon.bmp",0)
cv2.imshow('before',img)

#Nén ảnh
U_k,Z_k,X_mean = PCA(img,int(img.shape[1]*percent))
print('U_k=',U_k)
#print('Z_k=',Z_k)
#Lưu dữ liệu vào file pkl
#U_k=U_k*1000
data=(U_k.astype(np.float16),Z_k.astype(np.float16),X_mean.astype(np.uint8))
file = open('PCA.pkl', 'wb')

pickle.dump(data, file)
file.close()
#Lấy dữ liệu từ file pkl
file = open('PCA.pkl', 'rb')
U_k, Z_k, X_mean = pickle.load(file)
file.close()
print('U_k_after=',U_k)
print('Z_k_after=',Z_k)
#Khôi phục ảnh
X_star= decode(U_k,Z_k,X_mean)
cv2.imshow('after',X_star.astype(np.uint8).T)
#cv2.waitKey(0)


#---------Tính tỉ số nén R---------
#B1: Lấy kích thước của file
import os
base_size=os.stat('pokemon.bmp').st_size
compress_size=os.stat('PCA.pkl').st_size

print(base_size)
print(compress_size)

#B2: Tính tỉ số nén, in ra tỉ số nén
print('K= ',percent,'R = ', compress_size/base_size)

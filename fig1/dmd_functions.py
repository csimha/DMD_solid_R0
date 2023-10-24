#---------------------------------------
# DMD functions 
#---------------------------------------

import numpy as np
import pandas as pd
import scipy as sci 

def dmd_func(x0, y0, end_frame, dt, path):
 
 Xi = []
 Xtemp = np.concatenate([x0, y0])
 Xi.append(Xtemp)  


#----read in the displacement files 
 for id in range(1, end_frame):

  infile_name = path + str(id) + '.txt'
 
  df = pd.read_csv(infile_name, delim_whitespace=True, header=None, skiprows=2)

  dfvalues = df.values

  u = dfvalues[:, 0]
  v = dfvalues[:, 1]


  Xtemp = np.concatenate([x0+u , y0+v])

  Xi.append(Xtemp)
  
 Xi = np.array(Xi)
 Xi = Xi.T

 Xi = Xi #- np.mean(Xi)
 
  
 # create DMD input-output matrices
 X = Xi[:,:-1]
 Y = Xi[:,1:]



 U2,Sig2,Vh2 = sci.linalg.svd(X , lapack_driver='gesvd')
 xfinal = x0 + u; yfinal = y0 + v;
 
 return [Xi, U2, Sig2, Vh2, X, Y, xfinal, yfinal]




#---------------------------------
#  DMD reconstruction

def dmd_reconstruction(r, U2, Y, Vh2, Sig2, Xi, frame_num, xfinal, yfinal):

 
 U   = U2[:,:r]
 Sig = np.diag(Sig2)[:r,:r]
 V   = Vh2.conj().T[:,:r]
   
 Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.inv(Sig))

 mu,W = sci.linalg.eig(Atil)

 Xi_test = Xi[:, 0]    #----this is the first snapshot
 print(np.shape(Xi),np.shape(U), np.shape(Xi_test))
 for i in range(frame_num):
    Xi_test_r  = np.dot(U.T, Xi_test)
    Xi_tilda   = np.dot(Atil,Xi_test_r)
    Xi_appr    = np.dot(U,Xi_tilda)
    Xi_test    = Xi_appr
    
 node_num  = int(len(Xi[:, 0])/2)
 print(node_num)
 x0 = Xi[0: node_num, 0]
 y0 = Xi[node_num:2*node_num, 0]
 xfin_p = Xi_test[0: len(x0)]
 yfin_p = Xi_test[len(x0) : 2*len(x0)]    
 error_norm = np.sqrt(  (xfinal - xfin_p)**2 + (yfinal - yfin_p)**2)

 return error_norm
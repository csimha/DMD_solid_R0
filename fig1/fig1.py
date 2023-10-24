#-----------------------------------------------------------
# figure 1 plot 
# C. Hari Manoj Simha - University of Guelph - 2023
#  import modules
#  gavish_donoho is a module for rank truncation (function svht)
#  dmd_functions has functions used by this script
#  fe_mesh is for plotting abaqus mesh
#  connect.txt  - output from abaqus with element connectiveity list
#  initcoord.txt - output from abaqus with initial location of nodal coordinates
#-------------------------------------------------------------
import numpy as np

import matplotlib.pyplot as plt


import matplotlib.ticker as ticker

import matplotlib.tri as tri
from matplotlib.pyplot import *
matplotlib.rcdefaults()  #---reset to defaults
#-------------------------------------------------------


from matplotlib.pyplot import rcParams
rcParams['xtick.labelsize']  = 8
rcParams['ytick.labelsize']  = 8
rcParams['legend.fontsize']  = 8
rcParams['axes.titlesize']   = 8
rcParams['axes.labelsize']   = 8

#-----------------------------------------
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rcParams['ps.useafm'] = True
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

#-----------------------------------------
# import gavish donoho functions


from gavish_donoho import *

#---------------------------------------------------
# import element connnectivey and coord list fucntions

from fe_mesh import *
from dmd_functions import * 


elems = connect('connect.txt')
[x0, y0] = init_coords('initcoord.txt')

#-----subplot parameters
fig1 = plt.figure()

ax=grid = plt.GridSpec(9,3 )
ax.update(wspace=0.5, hspace=0.25)    #---space

#----element list
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 

#---common variables
dt = 0.01
path = 'disp'
#-------------------------------------------------------
#  plots for full reconstruction  and mesh
#-------------------------------------------------------
end_frame = 50

[Xi, U2, Sig2, Vh2, X, Y, xfinal, yfinal] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function

[rnk, tau ]= svht(X, sv=Sig2)                             #---rnk in Gavish Donoho

print(rnk)
    
error_norm =     dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  49 , xfinal, yfinal)


#----mesh plot
#------mesh
triangulation = tri.Triangulation(xfinal, yfinal, yy)
ax = plt.subplot(grid[0:5,0:1], aspect=.85)
ax.triplot(triangulation, '-k', linewidth= 0.75)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('x [mm]')
ax.set_ylabel('y [mm]')

ax.set_title('(a)')

#---------------error plots
ele = 22; azim = -43

ax1 =plt.subplot(grid[0:5, 1:3], projection='3d')
ax1.plot_trisurf(triangulation, error_norm, cmap='Greys_r')
ax1.set_xlabel('x [mm]')
ax1.set_ylabel('y [mm]')
ax1.set_zlabel('Error [mm]')
ax1.set_title('(b) $m =$'+ str(end_frame) + ', $r$ = ' +str(rnk))
ax1.grid(False)
ax1.view_init(ele, azim)
ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

#-------------------------------------------------------
#           Unit circle and mode plots
#-------------------------------------------------------
# Unit Circle for Atilde matrix - assume end_frame = 20
#-------------------------------------------------------

rcParams['xtick.labelsize']  = 8
rcParams['ytick.labelsize']  = 8
rcParams['legend.fontsize']  = 8
rcParams['axes.titlesize']   = 8
rcParams['axes.labelsize']   = 8


ax2 = plt.subplot(grid[6:9,0])

r = rnk
U = U2[:,:r]
Sig = np.diag(Sig2)[:r,:r]
V = Vh2.conj().T[:,:r]

Atil = np.dot(np.dot(np.dot(U.conj().T, Y), V), np.linalg.inv(Sig))
mu,W = eig(Atil)

wre = np.real(mu)
wim = np.imag(mu)

th = np.linspace(0, 2.*np.pi, 100)
xunit = np.cos(th); yunit = np.sin(th)


ax2.plot(xunit, yunit, '-k', linewidth=0.75)
ax2.set_aspect('equal', 'box')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.plot(wre, wim, 'sk', markerfacecolor='w', markersize=4)
ax2.set_xlabel('$Re(\lambda)$')
ax2.set_ylabel('$Im(\lambda)$')


#-------------------------------------------------------
# mode plots - assume end_frame = 50
#-------------------------------------------------------

ts = np.linspace(0, dt*end_frame, end_frame)
# build DMD modes
Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W)
b = np.dot(np.linalg.pinv(Phi), X[:,0])
Psi = np.zeros([rnk, len(ts)], dtype='complex')



for i,_ts in enumerate(ts):
    Psi[:,i] = np.multiply(np.power(mu, _ts/dt), b)
    

for i in range(rnk):
 ax2 = plt.subplot(grid[6+i, 1])
 
 max_psi = max(Psi[i, :])
 max_psi = np.sign(max(Psi[i, :]))*max_psi


 ax2.plot(ts, Psi[i, :], '--k', linewidth=0.75)
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)

 if ( i<rnk-1):
  ax2.set_xticks([])
 if ( i==0):
  ax2.set_title('$\Psi(t)$', fontsize=12) 

ax2.set_xlabel('Time')


#-------------------------------------------------------
# Amplitude plots - assume end_frame = 50
#-------------------------------------------------------


for i in range(rnk):
 ax2 = plt.subplot(grid[6+i, 2])
 ax2.plot(Phi[0:20, i],  '--k' , linewidth=0.75)
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)
 if ( i<rnk-1):
  ax2.set_xticks([])
  
 if ( i==0):
  ax2.set_title('$\Phi$', fontsize=12)
ax2.set_xlabel('Node #')


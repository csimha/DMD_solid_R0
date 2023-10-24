#-----------------------------------------------------------
# figure 2 plot 
# from 
# Dynamic Mode Decomposition of Deformation Fields in Elastic and Elastic-Plastic
# Solids 
# European Journal of Solid Mechanics A
# C. Hari Manoj Simha - University of Guelph - 2023
#  import modules
#  gavish_donoho is a module for rank truncation (function svht)
#  dmd_functions has functions used by this script
#  fe_mesh is for plotting abaqus mesh
#  connect.txt  - output from abaqus with element connectiveity list
#  initcoord.txt - output from abaqus with initial location of nodal coordinates
#-------------------------------------------------------------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.ticker as ticker
import matplotlib.tri as tri
from matplotlib.pyplot import *
matplotlib.rcdefaults()  #---reset to defaults
#-------------------------------------------------------

rcParams['figure.figsize']   = [8, 4.94]
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
import sys
from gavish_donoho import *

#---------------------------------------------------
# import element connnectivey and coord list fucntions

from fe_mesh import *
from dmd_functions import * 


elems = connect('connect.txt')
[x0, y0] = init_coords('initcoord.txt')

#-----subplot parameters
fig1 = plt.figure()

ax1=grid = plt.GridSpec(10, 6 )
ax1.update(wspace=0.5, hspace=0.55)    #---space

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

[rnk, tau ]= svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

print(rnk)
    
error_norm =     dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  49 , xfinal, yfinal)



#----mesh plot
#------mesh
triangulation = tri.Triangulation(xfinal, yfinal, yy)

#---------------error plots
ele = 22; azim = -73

#-------------------------------------------------------
#  plots for prediction  using frame 20 
#-------------------------------------------------------
end_frame = 20
[Xi, U2, Sig2, Vh2, X, Y, x15, y15] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function

[rnk, tau ]                         = svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

error_norm                          =  dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  49 , xfinal, yfinal)

print(rnk)

ax1 =plt.subplot(grid[0:5, 0:2], projection='3d')
ax1.plot_trisurf(triangulation, error_norm, cmap='Greys_r')
#ax1.set_xlabel('x [mm]')
#ax1.set_ylabel('y [mm]')
#ax1.set_zlabel('Error [mm]')

ax1.set_title( '(a) $m =$' + str(end_frame) + ', $r$ = ' +str(rnk))
ax1.grid(False)
ax1.view_init(ele, azim)
ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
#-------------------------------------------------------
#  plots for prediction  using frame 15 
#-------------------------------------------------------
end_frame = 15
[Xi, U2, Sig2, Vh2, X, Y, x15, y15] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function

[rnk, tau ]= svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

error_norm =     dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  49 , xfinal, yfinal)

print(rnk)

ax1 =plt.subplot(grid[0:5, 3:5], projection='3d')
ax1.plot_trisurf(triangulation, error_norm, cmap='Greys_r')


ax1.set_title('(b) $m =$'+ str(end_frame) + ', $r$ = ' +str(rnk))
ax1.grid(False)
ax1.view_init(ele, azim)

ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

#-------------------------------------------------------
#           Unit circle and mode plots
#-------------------------------------------------------
# Unit Circle for Atilde matrix - assume end_frame = 20
#-------------------------------------------------------
ax2 = plt.subplot(grid[5:9, 0:1])

rcParams['xtick.labelsize']  = 8
rcParams['ytick.labelsize']  = 8
rcParams['legend.fontsize']  = 8
rcParams['axes.titlesize']   = 8
rcParams['axes.labelsize']   = 8

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
ax2.set_title('(d) $m =$'+ str(end_frame) + ', $r$ = ' +str(rnk))

#-------------------------------------------------------
# Rank truncation plots 
#-------------------------------------------------------

ax2= plt.subplot(grid[5:9, 2:3])

ax2.plot(np.log(Sig2/sum(Sig2)), 'ok', markersize=4, markerfacecolor='w')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlabel('Modes')

ax2.set_xticks([0, 5, 10, 15, 20])

ax2.set_title('(c) $m =$'+ str(end_frame) + ', $r$ = ' +str(rnk))

ax2.set_ylabel(r'$log(\lambda_i/\sum_i^r \lambda_i)$' )
#-------------------------------------------------------
# mode plots - assume end_frame = 15
#-------------------------------------------------------

ts = np.linspace(0, dt*end_frame, end_frame)
# build DMD modes
Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W)
b = np.dot(np.linalg.pinv(Phi), X[:,0])
Psi = np.zeros([rnk, len(ts)], dtype='complex')


for i,_ts in enumerate(ts):
    Psi[:,i] = np.multiply(np.power(mu, _ts/dt), b)
    
#fig3 , ax3 = plt.subplots()


for i in range(rnk):
 ax2 = plt.subplot(grid[5+ i, 4:5])
 
 max_psi = max(Psi[i, :])
 max_psi = np.sign(max(Psi[i, :]))*max_psi
 max_psi = 1.

 ax2.plot(ts, Psi[i, :]/max_psi, '--k', linewidth=0.75)
 ax2.spines['top'].set_visible(False)
 ax2.spines['right'].set_visible(False)
 if (i==2):
  ax2.ticklabel_format(axis='both', style='sci', scilimits=(-3,3))
  
 if ( i<rnk-1):
  ax2.set_xticks([])
 #ax2.set_ylim([-1, 1])
 if (i==0):
  ax2.set_title('(e) ' + r'$\Psi(t)$')
ax2.set_xlabel('Time [s]')
 

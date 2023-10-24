#-----------------------------------------------------------
# figure 3 and 4 plot 
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

#-----------------------------------------
from matplotlib.pyplot import rcParams
from matplotlib import rc
#plt.rcParams['text.usetex'] = True
#plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
#-----------------------------------------
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rcParams['ps.useafm'] = True
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
from matplotlib.pyplot import rcParams
rcParams['xtick.labelsize']  = 8
rcParams['ytick.labelsize']  = 8
rcParams['legend.fontsize']  = 8
rcParams['axes.titlesize']   = 8
rcParams['axes.labelsize']   = 8
#rcParams['figure.subplot.bottom'] = 0.125
#rcParams['figure.subplot.right']  = 0.96

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

ax1=grid = plt.GridSpec(2,2 )
ax1.update(wspace=0.5, hspace=0.5)    #---space

#----element list
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 

#---common variables
dt = 0.01
path = 'disp'
#-------------------------------------------------------
#  plots for prediction using frame 6
#-------------------------------------------------------
end_frame = 7

[Xi, U2, Sig2, Vh2, X, Y, xfinal, yfinal] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function


[rnk, tau ]= svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

print(rnk)
 
error_norm =     dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  6 , xfinal, yfinal)


#---------------error plots
ele = 22; azim = -43
#------mesh
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 
triangulation = tri.Triangulation(xfinal, yfinal, yy)

ax1 =plt.subplot(grid[0, 1], projection='3d')
ax1.plot_trisurf(triangulation, error_norm, cmap='Greys_r')
#ax1.set_xlabel('x [mm]')
#ax1.set_ylabel('y [mm]')

ax1.set_title('(b) Error [mm] $m$ = '+ str(end_frame) + ', $r$ = ' +str(rnk))
ax1.grid(False)
ax1.view_init(ele, azim)
ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
#-------------------------------------------------------
#  plots for prediction using frame 7
#-------------------------------------------------------
end_frame = 8

[Xi, U2, Sig2, Vh2, X, Y, xfinal, yfinal] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function



[rnk, tau ]= svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

print(rnk)
    
error_norm =     dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  8 , xfinal, yfinal)


#---------------error plots
ele = 22; azim = -43
#------mesh
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 
triangulation = tri.Triangulation(xfinal, yfinal, yy)

ax1 =plt.subplot(grid[1, 0], projection='3d')
ax1.plot_trisurf(triangulation, error_norm, cmap='Greys_r')
#ax1.set_xlabel('x [mm]')
#ax1.set_ylabel('y [mm]')
ax1.ticklabel_format(style='sci',axis='z', scilimits=(0,0))

ax1.set_title( '(c) $m$ ='+ str(end_frame) + ', $r$ = ' +str(rnk))
ax1.grid(False)
ax1.view_init(ele, azim)
ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
#-------------------------------------------------------
#  plots for prediction using frame 8
#-------------------------------------------------------
end_frame = 12

[Xi, U2, Sig2, Vh2, X, Y, xfinal, yfinal] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function

[rnk, tau ]= svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

print(rnk)
    
error_norm =     dmd_reconstruction(rnk, U2, Y, Vh2, Sig2, Xi,  end_frame , xfinal, yfinal)


#---------------error plots
ele = 22; azim = -43
#------mesh
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 
triangulation = tri.Triangulation(xfinal, yfinal, yy)

ax1 =plt.subplot(grid[1, 1], projection='3d')
ax1.plot_trisurf(triangulation, error_norm, cmap='Greys_r')
#ax1.plot_trisurf(triangulation, error_norm, cmap='Greys')

#ax1.set_xlabel('x [mm]')
#ax1.set_ylabel('y [mm]')
ax1.ticklabel_format(style='sci', axis='z', scilimits=(0,0))

ax1.set_title( '(d) $m$ =' +str(end_frame) + ', $r$ = ' +str(rnk))
ax1.grid(False)
ax1.view_init(ele, azim)

ax1.zaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

#-------------------------------------------------------
#  force-displacement plots
#-------------------------------------------------------
force_df = pd.read_csv('force_disp.txt', delim_whitespace=True, header=0)
disp = force_df.values[:,0]
force = force_df.values[:,1]
time_force = force_df.values[:, 2]

#----force plot
ax1 = plt.subplot(grid[0,0])


ax1t =  ax1.twiny()
new_tick_locations = np.array([0, 0.005, 0.01, 0.015])

def tick_function(X):
    V = X/0.1
    return ["%.3f" % z for z in V]
ax1.set_xlim([0, .015])
ax1.plot(disp[0:16], force[0:16], '-k', linewidth=1.0)
ax1t.set_xlim(ax1.get_xlim())
ax1t.set_xticks(new_tick_locations)
ax1t.set_xticklabels(tick_function(new_tick_locations))


dummy_disp = np.array([0.0, 0.005, 0.017])
ax1.plot( dummy_disp, dummy_disp*134273.04195804194  + 4.009102564102591, '--k', linewidth= 1)
ax1.plot([0.008, 0.008], [0, 2000], ':k', linewidth=0.75)
ax1.plot([0.012, 0.012], [0, 2000], ':k', linewidth=0.75)
ax1.text( 0.006, 1950, r'$m=8$', fontsize=8)

ax1.text( 0.011, 1950, r'$m=12$', fontsize=8)


ax1.set_xlabel('Displacement [mm]')
ax1.set_ylabel('Load [N]')
ax1.set_title('(a)')
ax1t.set_xlabel('Time [s]')


#----------------------------------------------
# Figure 2

#-------------------------------------------------------
# Unit Circle for Atilde matrix - assume end_frame = 7
#-------------------------------------------------------

fig2 = plt.figure()

ax2=grid = plt.GridSpec(2,2 )
ax2.update(wspace=0.4, hspace=0.4)    #---space


rcParams['xtick.labelsize']  = 8
rcParams['ytick.labelsize']  = 8
rcParams['legend.fontsize']  = 8
rcParams['axes.titlesize']   = 8
rcParams['axes.labelsize']   = 8

end_frame = 8

[Xi, U2, Sig2, Vh2, X, Y, xfinal, yfinal] = dmd_func(x0, y0, end_frame, dt, path)  #---call DMD function

[rnk, tau ]= svht(Xi, sv=Sig2)                             #---rnk in Gavish Donoho

print(rnk)

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

ax2 =plt.subplot(grid[1, 0])
ax2.plot(xunit, yunit, '-k', linewidth=1.)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.plot(wre, wim, 'sk', markerfacecolor='w', markersize=6)
ax2.set_xlabel('$Re(\lambda)$')
ax2.set_ylabel('$Im(\lambda)$')

ax2.set_xlim([-2, 3])
ax2.set_ylim([-2, 3])
ax2.set_aspect('equal', 'box')


left, bottom, width, height = [0.33, 0.32, 0.1, 0.1]
ax5 = fig2.add_axes([left, bottom, width, height])
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.plot(wre, wim, 'sk', markerfacecolor='k', markersize=6)
ax5.set_xlim([15, 18])
ax5.set_yticks([0])
ax5.set_xticks([15, 18])

ax2.set_title('(c) $m$ = 8, $r$ = 3')
#-------------------------------------------------------
# mode plots - assume end_frame = 7
#-------------------------------------------------------

ts = np.linspace(0, dt*end_frame, end_frame)
# build DMD modes
Phi = np.dot(np.dot(np.dot(Y, V), np.linalg.inv(Sig)), W)
b = np.dot(np.linalg.pinv(Phi), X[:,0])
Psi = np.zeros([rnk, len(ts)], dtype='complex')

ax2 = plt.subplot(grid[1, 1])

for i,_ts in enumerate(ts):
    Psi[:,i] = np.multiply(np.power(mu, _ts/dt), b)
    


for i in range(rnk):
    plt.plot(ts, Psi[i, :], '--k', linewidth=1.0)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_ylabel(r'$\Psi$', fontsize=12)
ax2.set_xlabel('Time [s]')
ax2.set_title('(d) $\Phi(t)$ m= 8')

#------------------------------------------------------

#-------------------------------------------------------
# meshplots  frame 8
#-------------------------------------------------------
elems = connect('connect.txt')
[x0, y0] = init_coords('initcoord.txt')
#----element list
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 

#----mesh plot
#------mesh
triangulation = tri.Triangulation(xfinal, yfinal, yy)
ax2 = plt.subplot(grid[0,0], aspect=.85)
ax2.triplot(triangulation, '-k', linewidth= 0.15)
#ax2.axis('off')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim([0, 8])
ax2.set_ylim([0, 7 ])

#ax2.spines['left'].set_visible(False)
#ax2.spines['bottom'].set_visible(False)

acyield6 = pd.read_csv('acyield_6.txt', delim_whitespace=True, skiprows=0, header=0)

ac6 = acyield6.values[:,1]
nodenum = acyield6.values[:, 0]
nodenum = nodenum - 1  #---to account for python arrays starting from 0
ac6_reorder = np.zeros(len(ac6))

for i in range(0, len(ac6) -1): 
 ac6_reorder[ int( nodenum[i])]  = ac6[i]
xx =ax2.tricontourf(triangulation, ac6_reorder, cmap='Greys_r')
#xx =ax2.tricontourf(triangulation, ac6_reorder, cmap='gray_r')


fig2.colorbar(xx, ticks= [0, 0.7])
ax2.set_title('(a) Yield Flag $m$ = 7')
#-----------------------------------------------------------
#-------------------------------------------------------
# meshplots  frame 8
#-------------------------------------------------------
elems = connect('connect.txt')
[x0, y0] = init_coords('initcoord.txt')
#----element list
yy = [[elem -1 for elem in sublist]
  for sublist in elems]  #---shift element numbers by 1 

#----mesh plot
#------mesh
triangulation = tri.Triangulation(xfinal, yfinal, yy)
ax2 = plt.subplot(grid[0,1], aspect=.85)
ax2.triplot(triangulation, '-k', linewidth= 0.15)
#ax2.axis('off')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_xlim([0, 8])
ax2.set_ylim([0, 7 ])

#ax2.spines['left'].set_visible(False)
#ax2.spines['bottom'].set_visible(False)

acyield6 = pd.read_csv('acyield_8.txt', delim_whitespace=True, skiprows=0, header=0)

ac6 = acyield6.values[:,1]
nodenum = acyield6.values[:, 0]
nodenum = nodenum - 1  #---to account for python arrays starting from 0
ac6_reorder = np.zeros(len(ac6))

for i in range(0, len(ac6) -1): 
 ac6_reorder[ int( nodenum[i])]  = ac6[i]
xx =ax2.tricontourf(triangulation, ac6_reorder, cmap='Greys_r')

fig2.colorbar(xx, ticks= range(0,2))

ax2.set_title('(b) Yield Flag $m$ = 8')
#-----------------------------------------------------------

import pandas as pd

#---------------------------------------
# function for connectivity list
#----------------------------------------

def connect(connect_fil):
 
 #----connect_fil is the name of the connectivity file list 
 con = pd.read_csv(connect_fil, delim_whitespace=True, header=None)


 elems = con.values.tolist()
 
 return elems


#-----------------------------------------
# function for coordinates of nodes
#----------------------------------------

def init_coords(init_file):
 
 df = pd.read_csv(init_file, delim_whitespace=True, header=None, skiprows=2)
 dfvalues = df.values
 x0 = dfvalues[:, 0]
 y0 = dfvalues[:, 1]
 
 return [x0, y0]
    
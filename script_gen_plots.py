from postrpocessing import render_pvd2file_2d_color
import os
filename = '../CahnHilliard/IC_TwoBubbles/CH_a_0.500_T_0.100000_nt_2000_Nx_16_e_0.1000sol.pvd'
# import pdb; pdb.set_trace()    
prefix,_ = os.path.splitext(filename)
render_pvd2file_2d_color(filename,1e-4, outputfile=prefix + '0', outputformat='pdf',datatitle='phi0',filetitle ='Initial value')
render_pvd2file_2d_color(filename,float("inf"), outputformat='pdf',datatitle='phi',filetitle ='Final value')

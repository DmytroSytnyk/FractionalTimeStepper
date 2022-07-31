from postrpocessing import render_pvd2file_2d_color
import os, glob
# filename = '../CahnHilliard/IC_TwoBubbles/CH_a_0.500_T_0.100000_nt_2000_Nx_16_e_0.1000sol.pvd'
# filename = '../CahnHilliard/IC_BubbleSoup/CH_a_0.500_T_2.000000_nt_10000_Nx_64_e_50.0000sol.pvd'
# import pdb; pdb.set_trace()    
filename =glob.glob('../CahnHilliard/*/*.pvd')
# import pdb; pdb.set_trace()    
for f in filename:
    prefix,_ = os.path.splitext(f)
    render_pvd2file_2d_color(f,1e-4, outputfile=prefix + '0', outputformat='pdf',datatitle='phi0',filetitle ='Initial value')
    render_pvd2file_2d_color(f,float("inf"), outputformat='pdf',datatitle='phi',filetitle ='Final value')

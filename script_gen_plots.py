from postrpocessing import render_pvd2file_2d_color
import os, glob
# path = '../!Results/CahnHilliard/*/*.pvd'
path = './CahnHilliard/*/*.pvd'
print('Filename search glob is ', path)
filename =glob.glob(path)
# import pdb; pdb.set_trace()    
for f in filename:
    print()
    print('Processing ', f)
    print('############################################')
    prefix,_ = os.path.splitext(f)
    # render_pvd2file_2d_color(f,1e-4, outputfile=prefix + '0', outputformat='pdf',datatitle='phi0',filetitle ='Initial value')
    # render_pvd2file_2d_color(f,float("inf"), outputformat='pdf',datatitle='phi',filetitle ='Final value')
    render_pvd2file_2d_color(f,"all", outputformat='pdf',datatitle='phi')



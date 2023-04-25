
import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

size = int(35)

arch = [ 
    to_head('..'), 
    to_cor(),
    to_begin(),
    to_input( ' id01_18_origin.png' ),

    #block-001
    to_ConvConvRelu( name='ccr_b1', s_filer=256, n_filer=(64,64), offset="(0,0,0)", to="(0,0,0)", width=(2,2), height=size, depth=size , caption="U-Net" ),
    to_Pool(name="pool_b1", offset="(0,0,0)", to="(ccr_b1-east)", width=1, height=size-7, depth=size-7, opacity=0.5),
    
    to_Conv("conv1", 256, 64, offset="(0.5,0,0)", to="(pool_b1-east)",  height=size-30, depth=size-30, width=6 ,fill=2,caption="ResNet50"),
    to_connection( "pool_b1", "conv1"),
    
    *block_2ConvPool( name='b2', botton='conv1', top='pool_b2', s_filer=128, n_filer=128, offset="(1,0,0)", size=(size-7,size-7,3.5), opacity=0.5 ),
    to_Conv("conv2", 256, 64, offset="(0.5,0,0)", to="(pool_b2-east)",  height=size-30, depth=size-30, width=6 ,fill=2,caption="ResNet50"),
    to_connection( "pool_b2", "conv2"),
    
    *block_2ConvPool( name='b3', botton='conv2', top='pool_b3', s_filer=64, n_filer=256, offset="(1,0,0)", size=(size-24,size-24,4.5), opacity=0.5 ),
    to_Conv("conv3", 256, 64, offset="(0.5,0,0)", to="(pool_b3-east)",  height=size-30, depth=size-30, width=6 ,fill=2,caption="ResNet50"),
    to_connection( "pool_b3", "conv3"),
    
    *block_2ConvPool( name='b4', botton='conv3', top='pool_b4', s_filer=32,  n_filer=512, offset="(1,0,0)", size=(size-24,size-24,5.5), opacity=0.5 ),
    to_Conv("conv4", 256, 64, offset="(0.5,0,0)", to="(pool_b4-east)",  height=size-30, depth=size-30, width=6 ,fill=2,caption="ResNet50"),
    to_connection( "pool_b4", "conv4"),
    
    #Bottleneck
    #block-005
    to_ConvConvRelu( name='ccr_b5', s_filer=16, n_filer=(1024,1024), offset="(1,0,0)", to="(conv4-east)", width=(size-32,size-32), height=8, depth=8, caption="Bottleneck"  ),
    to_connection( "pool_b4", "ccr_b5"),

    #Decoder
    *block_Unconv( name="b6", botton="ccr_b5", top='end_b6', s_filer=32,  n_filer=512, offset="(1.5,0,0)", size=(size-24,size-24,5.0), opacity=0.5 ),
    to_skip( of='ccr_b4', to='ccr_res_b6', pos=1.25),
    *block_Unconv( name="b7", botton="end_b6", top='end_b7', s_filer=64, n_filer=256, offset="(1.5,0,0)", size=(size-24,size-24,4.5), opacity=0.5 ),
    to_skip( of='ccr_b3', to='ccr_res_b7', pos=1.75),    
    *block_Unconv( name="b8", botton="end_b7", top='end_b8', s_filer=128, n_filer=128, offset="(1.5,0,0)", size=(size-7,size-7,3.5), opacity=0.5 ),
    to_skip( of='ccr_b2', to='ccr_res_b8', pos=1.25),    
    
    *block_Unconv( name="b9", botton="end_b8", top='end_b9', s_filer=256, n_filer=64,  offset="(1.5,0,0)", size=(size,size,2.5), opacity=0.5 ),
    to_skip( of='ccr_b1', to='ccr_res_b9', pos=1.25),
    
    to_cor(),
    to_Conv("conv2", 256, 64, offset="(1,0,0)", to="(end_b9-east)", height=size, depth=size, width=2, caption="DnCNN",fill=2 ),
    to_Relu(name="pool_conv2", offset="(0,0,0)", to="(conv2-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "end_b9", "conv2"),
    
    to_Conv("conv3", 256, 64, offset="(1,0,0)", to="(pool_conv2-east)",  height=size, depth=size, width=2 ,fill=2),
    to_Relu(name="pool_conv3", offset="(0,0,0)", to="(conv3-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "conv2", "conv3"),
    
    to_Conv("conv4", 256, 64, offset="(1,0,0)", to="(pool_conv3-east)",  height=size, depth=size, width=2 ,fill=2),
    to_Relu(name="pool_conv4", offset="(0,0,0)", to="(conv4-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "conv3", "conv4"),
    
    to_ConvSoftMax( name="soft1", s_filer=256, offset="(0.75,0,0)", to="(conv4-east)", width=1, height=size, depth=size, caption="Sigmoid" ),
    to_connection( "conv4", "soft1"),
    
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    

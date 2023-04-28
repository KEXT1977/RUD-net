
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
    to_Conv("conv2", 256, 64, offset="(0.5,0,0)", height=size, depth=size, width=5, caption="DnCNN",fill=2 ),
    to_Relu(name="pool_conv2", offset="(0,0,0)", to="(conv2-east)", width=1, height=size, depth=size, opacity=0.5),
    
    to_Conv("conv3", 256, 64, offset="(1.5,0,0)", to="(pool_conv2-east)",  height=size, depth=size, width=5 ,fill=2),
    to_Conv("conv35", 256, 64, offset="(0,0,0)", to="(conv3-east)",  height=size, depth=size, width=5 ,fill=1),
    to_Relu(name="pool_conv3", offset="(0,0,0)", to="(conv35-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "pool_conv2", "conv3"),
    
    to_Conv("conv4", 256, 64, offset="(1.5,0,0)", to="(pool_conv3-east)",  height=size, depth=size, width=5 ,fill=2),
    to_Conv("conv45", 256, 64, offset="(0,0,0)", to="(conv4-east)",  height=size, depth=size, width=5 ,fill=1),
    to_Relu(name="pool_conv4", offset="(0,0,0)", to="(conv45-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "pool_conv3", "conv4"),
    
    to_Conv("conv5", 256, 64, offset="(1.5,0,0)", to="(pool_conv4-east)",  height=size, depth=size, width=5 ,fill=2),
    to_Conv("conv55", 256, 64, offset="(0,0,0)", to="(conv5-east)",  height=size, depth=size, width=5 ,fill=1),
    to_Relu(name="pool_conv5", offset="(0,0,0)", to="(conv55-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "pool_conv4", "conv5"),
    
    to_Conv("conv6", 256, 64, offset="(1.5,0,0)", to="(pool_conv5-east)",  height=size, depth=size, width=5 ,fill=2),
    to_Conv("conv65", 256, 64, offset="(0,0,0)", to="(conv6-east)",  height=size, depth=size, width=5 ,fill=1),
    to_Relu(name="pool_conv6", offset="(0,0,0)", to="(conv65-east)", width=1, height=size, depth=size, opacity=0.5),
    to_connection( "pool_conv5", "conv6"),
    
    to_ConvSoftMax( name="soft1", s_filer=256, offset="(2.55,0,0)", to="(conv6-east)", width=1, height=size, depth=size, caption="Sigmoid" ),
    to_connection( "pool_conv6", "soft1"),
    
    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    

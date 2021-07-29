from torch import nn
import torch
from torch.autograd import Variable


class DoubleAttentionLayer(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1,concat=True ):
        super(DoubleAttentionLayer, self).__init__()

        self.K           = k
        self.c_m = c_m
        self.c_n = c_n
        self.softmax     = nn.Softmax(dim=1)
        self.in_channels = in_channels

        self.convA  = nn.Conv2d(in_channels, c_m, 1)
        self.convB  = nn.Conv2d(in_channels, c_n, 1)
        self.convV  = nn.Conv2d(in_channels, c_n, 1)
        #self.bnZ    = nn.BatchNorm2d(c_m)

        # self.convExtension = nn.Conv2d(c_m,in_channels,1)
        # self.bnZ    = nn.BatchNorm2d(in_channels)
        # self.relu   = nn.ReLU()

        self.concat = concat

    def forward(self, x):

        b, c, h, w = x.size()

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        tmpA = A.view( batch, self.K, self.c_m, h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*h*w )
        tmpB = B.view( batch, self.K, self.c_n, h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*h*w )
        tmpV = V.view( batch, self.K, self.c_n, h*w ).permute(0,1,3,2).contiguous().view( int(b*h*w), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB )  #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV )  #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view(batch, self.c_m, self.K,h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, h, w )
        #tmpZ     = self.bnZ(tmpZ)

        # tmpE     = self.convExtension(tmpZ)
        # tmpE     = self.bnZ(tmpE)
        # tmpE     = self.relu(tmpE)

        if self.concat:
            return tmpZ+x
        else:
            return tmpZ



class DoubleAttentionLayer_norm(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1,concat=True ):
        super(DoubleAttentionLayer_norm, self).__init__()

        self.K           = k
        self.c_m         = c_m
        self.c_n         = c_n
        self.softmax     = nn.Softmax(dim=1)
        self.in_channels = in_channels

        self.convA  = nn.Conv2d(in_channels, c_m, 1)
        self.convB  = nn.Conv2d(in_channels, c_n, 1)
        self.convV  = nn.Conv2d(in_channels, c_n, 1)
        self.convExtension = nn.Conv2d(c_m,in_channels,1)
        self.bnZ    = nn.BatchNorm2d(in_channels)
        self.relu   = nn.ReLU()
        self.concat = concat

    def forward(self, x):

        b, c, h, w = x.size()

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        tmpA = A.view( batch, self.K, self.c_m, h*w ).permute(0,2,1,3).view( batch, self.c_m, self.K*h*w )
        tmpB = B.view( batch, self.K, self.c_n, h*w ).permute(0,2,1,3).view( batch*self.c_n, self.K*h*w )
        tmpV = V.view( batch, self.K, self.c_n, h*w ).permute(0,1,3,2).contiguous().view( int(b*h*w), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*h*w ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*h*w, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB )  #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV )  #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view( batch, self.c_m, self.K, h*w).permute( 0, 2, 1,3).view( int(b), self.c_m, h, w )
        tmpE     = self.convExtension(tmpZ)
        tmpE     = self.bnZ(tmpE)
        tmpE     = self.relu(tmpE)

        if self.concat:
            return tmpE+x
        else:
            return tmpE



class Double_DoubleAttentionLayer_norm(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1,concat=True ):
        super(Double_DoubleAttentionLayer_norm, self).__init__()

        self.K           = k
        self.c_m         = c_m
        self.c_n         = c_n
        self.softmax     = nn.Softmax(dim=1)
        self.in_channels = in_channels

        #####
        self.convA  = nn.Conv2d(in_channels, c_m, 1)
        self.convB  = nn.Conv2d(in_channels, c_n, 1)
        self.convV  = nn.Conv2d(in_channels, c_n, 1)
        self.convExtension = nn.Conv2d(c_m,in_channels,1)
        self.bnZ    = nn.BatchNorm2d(in_channels)
        self.relu   = nn.ReLU()

        #####
        self.convA_S  = nn.Conv2d(in_channels, c_m, 1)
        self.convB_S  = nn.Conv2d(in_channels, c_m, 1)
        self.convV_S  = nn.Conv2d(in_channels, c_m, 1)
        self.convExtension_S = nn.Conv2d(c_m,in_channels,1)
        self.bnZ_S    = nn.BatchNorm2d(in_channels)
        self.relu_S   = nn.ReLU()

        self.conv_out = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bnZ_out  = nn.BatchNorm2d(in_channels)
        self.relu_out = nn.ReLU()

        self.concat   = concat

    def forward(self, x):

        b, c, h, w = x.size()

        assert c == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b/self.K)

        #################################################
        tmpA = A.view( batch, self.K, self.c_m, h*w ).permute(0,2,1,3).contiguous().view( batch, self.c_m, self.K*h*w )
        tmpB = B.view( batch, self.K, self.c_n, h*w ).permute(0,2,1,3).contiguous().view( batch*self.c_n, self.K*h*w )
        tmpV = V.view( batch, self.K, self.c_n, h*w ).permute(0,1,3,2).contiguous().view( int(b*h*w), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*h*w ).permute( 0, 2, 1).contiguous()  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*h*w, self.c_n ).permute( 0, 2, 1).contiguous()  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB )  #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV )  #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view( batch, self.c_m, self.K, h*w).permute( 0, 2, 1,3).contiguous().view( int(b), self.c_m, h, w )
        tmpE     = self.convExtension(tmpZ)
        tmpE     = self.bnZ(tmpE)
        tmpE     = self.relu(tmpE)

        ################################################
        A_S    = self.convA_S(x)
        B_S    = self.convB_S(x)
        V_S    = self.convV_S(x)
        tmpA_S = A_S.view( batch, self.K, self.c_m, h*w ).permute(0,3,1,2).contiguous().view( batch, h*w, self.K*self.c_m ) #batch, h*w, self.K*self.c_m
        # print(batch,self.K,self.c_m,h*w)
        # print(B_S.shape)
        tmpB_S = B_S.view( batch, self.K, self.c_m, h*w ).permute(0,3,1,2).contiguous().view( batch*h*w,  self.K*self.c_m ) #batch*h*w,  self.K*self.c_m
        tmpV_S = V_S.view( batch, self.K, self.c_m, h*w ).permute(0,3,1,2).contiguous().view( batch*h*w, self.K*self.c_m )  #batch*h*w,  self.K*self.c_m

        softmaxB_S = self.softmax(tmpB_S).view( batch, h*w, self.K*self.c_m ).permute( 0, 2, 1).contiguous()   #batch, self.K*self.c_m, h*w
        softmaxV_S = self.softmax(tmpV_S).view( batch, h*w, self.K*self.c_m )                                  #batch, h*w, self.K*self.c_m

        tmpG_S     = tmpA_S.matmul( softmaxB_S )  #batch, h*w, h*w
        tmpZ_S     = tmpG_S.matmul( softmaxV_S )  #batch, h*w, self.K*self.c_m
        tmpZ_S     = tmpZ_S.permute( 0, 2, 1).contiguous().view( int(b), self.c_m, h, w )
        tmpE_S     = self.convExtension(tmpZ_S)
        tmpE_S     = self.bnZ(tmpE_S)
        tmpE_S     = self.relu(tmpE_S)

        tmp_plus   = tmpE + tmpE_S + x
        tmp_out    = self.conv_out(tmp_plus)

        return tmp_out


class DoubleAttentionLayerBiBranch(nn.Module):
    def __init__(self, in_channels, c_m, c_n,k =1,concat=True ):
        super(DoubleAttentionLayerBiBranch, self).__init__()

        self.K           = k
        self.c_m = c_m
        self.c_n = c_n
        self.softmax     = nn.Softmax(dim=1)
        self.in_channels = in_channels

        self.convA  = nn.Conv2d(in_channels, c_m, 1)
        self.convB  = nn.Conv2d(in_channels, c_n, 1)
        self.convV  = nn.Conv2d(in_channels, c_n, 1)
        self.concat = concat

    def forward(self, x,y):

        b1, c1, h1, w1 = x.size()
        b2, c2, h2, w2 = y.size()

        assert c1 == self.in_channels,'input channel not equal!'
        #assert b//self.K == self.in_channels, 'input channel not equal!'

        A = self.convA(x)
        B = self.convB(x)
        V = self.convV(x)

        batch = int(b1/self.K)

        tmpA = A.view( batch, self.K, self.c_m, h1*w1 ).permute(0,2,1,3).view( batch, self.c_m, self.K*h1*w1 )
        tmpB = B.view( batch, self.K, self.c_n, h1*w1 ).permute(0,2,1,3).view( batch*self.c_n, self.K*h1*w1 )
        tmpV = V.view( batch, self.K, self.c_n, h1*w1 ).permute(0,1,3,2).contiguous().view( int(b1*h1*w1), self.c_n )

        softmaxB = self.softmax(tmpB).view( batch, self.c_n, self.K*h1*w1 ).permute( 0, 2, 1)  #batch, self.K*h*w, self.c_n
        softmaxV = self.softmax(tmpV).view( batch, self.K*h1*w1, self.c_n ).permute( 0, 2, 1)  #batch, self.c_n  , self.K*h*w

        tmpG     = tmpA.matmul( softmaxB ) #batch, self.c_m, self.c_n
        tmpZ     = tmpG.matmul( softmaxV ) #batch, self.c_m, self.K*h*w
        tmpZ     = tmpZ.view(batch, self.c_m, self.K,h1*w1).permute( 0, 2, 1,3).view( int(b1), self.c_m, h1, w1 )

        if self.concat:
            return tmpZ+y
        else:
            return tmpZ

if __name__ == "__main__":


    # tmp1        = torch.ones(2,2,3)
    # tmp1[1,:,:] = tmp1[1,:,:]*2
    # tmp2 = tmp1.permute(0,2,1)
    # print(tmp1)
    # print( tmp2)
    # print( tmp1.matmul(tmp2))

    in_channels = 10
    c_m = 4
    c_n = 3

    doubleA = DoubleAttentionLayer(in_channels, in_channels, c_n)

    x   = torch.ones(2,in_channels,6,8)
    x   = Variable(x)
    tmp = doubleA(x)

    print("result")

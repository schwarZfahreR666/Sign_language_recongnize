import torch
import torch.nn as nn
#from torchsummary import summary

import ds_mfnet
import initialize_ds

ofmodel_path = 'models/trained_model_of.pkl'
mfmodel_path = 'models/trained_model1.pkl'
class DS_NET(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes,batch_size, pretrained=False):
        super(DS_NET, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.batch_size = batch_size

        self.of_net = ds_mfnet.MFNET_3D(num_classes=self.num_classes,batch_size=self.batch_size)


        self.rgb_net = ds_mfnet.MFNET_3D(num_classes=self.num_classes,batch_size=self.batch_size)


        self.classifier = nn.Linear(768*2, self.num_classes)


        if self.pretrained:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            no_pre_params = initialize_ds.init_from_dict(self.rgb_net,mfmodel_path,device,Flags=True)
            
            

            initialize_ds.init_from_dict(self.of_net,ofmodel_path,device,Flags=True)

            print('loaded pretrained model')
        


    def forward(self, x,x_o):

        h=self.rgb_net(x)

        m=self.of_net(x_o)

        mh= torch.cat([h,m],1)
        mh = self.classifier(mh)

        return mh

    

if __name__ == "__main__":
    inputs = torch.rand(2, 3, 16, 224, 224)
    inputs_o = torch.rand(2,3,16,112,112)
    net = DS_NET(num_classes=100,batch_size=2,pretrained=True)

    outputs = net(inputs,inputs_o)
    outputs = outputs.view(outputs.shape[0],-1)
    print(outputs.size())
    #summary(net,input_size=(3,16,112,112))

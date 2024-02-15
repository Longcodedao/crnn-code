import torch.nn as nn
import torch
import torch.nn.functional as F


class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_classes,
                leaky_relu = True, map_to_seq = 64, lstm_hidden = 256):
        super(CRNN, self).__init__()
        self.cnn, dimension = self._cnn_backbone(img_channel, img_width, 
                                                 img_height, leaky_relu)
        output_channel, output_height, output_width = dimension

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq)
        self.lstm1 = nn.LSTM(map_to_seq, lstm_hidden, 
                            bidirectional = True, batch_first = True)
        self.lstm2 = nn.LSTM(2 * lstm_hidden, lstm_hidden, 
                            bidirectional = True, batch_first = True)

        self.dense = nn.Linear(2 * lstm_hidden, num_classes)
        
    def _cnn_backbone(self, img_channel, img_width, img_height,
                      leaky_relu = True):
        # assert img_width & 4 == 0
        # assert img_height % 16 == 0

        # m means the mode: 0 is convolution, 1 is max pooling
        # k means kernel_size
        # s means stride
        # p means padding
        cfgs = [
            #m   #k,     #s    #p    #c    #bn
            [0,  (3, 3),  1,   1,    64,  False],
            [1,  (2, 2),  2,   None, None, False],
            [0,  (3, 3),  1,   1,    128, False],
            [1,  (2, 2),  2,   None, None, False],
            [0,  (3, 3),  1,   1,    256,  False],
            [0,  (3, 3),  1,   1,    256,  False],
            [1,  (2, 1),  2,   None,  None, False],
            [0,  (3, 3),  1,   1,    512,   True],
            [0,  (3, 3),  1,   1,    512,   True],
            [1,  (2, 1),  2,   None,  None, False],
            [0,  (2, 2),  1,   0,    512,   False],
       ]

        cnn = []
        input_channels = img_channel
        output_channels = None

        for m, k, s, p, c, bn in cfgs:
            if m == 0: # Convolution 
                output_channels = c
                
                cnn.append(nn.Conv2d(input_channels, output_channels, 
                                     kernel_size = k, stride = s, 
                                     padding = p))
                relu = nn.LeakyReLU(0.2, inplace = True) if leaky_relu == True \
                            else nn.ReLU(inplace = True)
                cnn.append(relu)
                
                if bn == True:
                    cnn.append(nn.BatchNorm2d(output_channels))
                           
                input_channels = output_channels
                
            elif m == 1:
                cnn.append(nn.MaxPool2d(kernel_size = k, stride = s))

        cnn_module = nn.Sequential(*cnn)
        # The output height and width of an image after passing through CNN
        output_height = img_height // 16 - 1 
        output_width = img_width // 4 - 1
        
        return cnn_module, (output_channels, output_height, output_width)
    
    def forward(self, images):

        conv = self.cnn(images)
        batch, channels, height, width = conv.size()
        # print(f"Channels: {channels} Height: {height} Width: {width}")
        conv = conv.view(batch, channels * height, width)
        
        # (batch, width, lol)
        conv = conv.permute(0, 2, 1)
        print(conv.shape)
        
        sequence = self.map_to_seq(conv)

        recurrent, _ = self.lstm1(sequence)
        recurrent, _ = self.lstm2(recurrent)

        logits = self.dense(recurrent)

        # Returning the dimension (batch_size, width, num_classes)
        return F.log_softmax(logits, dim = 2)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





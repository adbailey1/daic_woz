import math
import torch
import torch.nn as nn


def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_lstm(layer):
    """
    Initialises the hidden layers in the LSTM - H0 and C0.

    Input
        layer: torch.Tensor - The LSTM layer
    """
    n_i1, n_i2 = layer.weight_ih_l0.size()
    n_i = n_i1 * n_i2

    std = math.sqrt(2. / n_i)
    scale = std * math.sqrt(3.)
    layer.weight_ih_l0.data.uniform_(-scale, scale)

    if layer.bias_ih_l0 is not None:
        layer.bias_ih_l0.data.fill_(0.)

    n_h1, n_h2 = layer.weight_hh_l0.size()
    n_h = n_h1 * n_h2

    std = math.sqrt(2. / n_h)
    scale = std * math.sqrt(3.)
    layer.weight_hh_l0.data.uniform_(-scale, scale)

    if layer.bias_hh_l0 is not None:
        layer.bias_hh_l0.data.fill_(0.)


def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock1d(nn.Module):
    """
    Creates an instance of a 1D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, np, normalisation):
        super(ConvBlock1d, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=np[0],
                               stride=np[1],
                               padding=np[2],
                               bias=False)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm1d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm == 'bn':
            x = self.relu(self.bn1(self.conv1(x)))
        else:
            x = self.relu(self.conv1(x))

        return x


class ConvBlock(nn.Module):
    """
    Creates an instance of a 2D convolutional layer. This includes the
    convolutional filter but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, np, normalisation, att=None):
        super(ConvBlock, self).__init__()
        self.norm = normalisation
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=np[0],
                               stride=np[1],
                               padding=np[2],
                               bias=False)
        if self.norm == 'bn':
            self.bn1 = nn.BatchNorm2d(out_channels)
        elif self.norm == 'wn':
            self.conv1 = nn.utils.weight_norm(self.conv1, name='weight')
        else:
            self.conv1 = self.conv1
        self.att = att
        if not self.att:
            self.act = nn.ReLU()
        else:
            self.norm = None
            if self.att == 'softmax':
                self.act = nn.Softmax(dim=-1)
            elif self.att == 'global':
                self.act = None
            else:
                self.act = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.conv1)
        else:
            init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input):
        """
        Passes the input through the convolutional filter

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        x_shape = x.shape
        if self.att:
            x = self.conv1(x)
            # x = x.reshape(x_shape[0], -1)
            if self.act():
                x = self.act(x)
        else:
            if self.norm == 'bn':
                x = self.act(self.bn1(self.conv1(x)))
            else:
                x = self.act(self.conv1(x))

        d = x.dim()
        if d == 4:
            batch, freq, height, width = x.shape
            if height == 1:
                x = x.reshape(batch, freq, width)
            elif width == 1:
                x = x.reshape(batch, freq, height)
        elif d == 3:
            batch, freq, width = x.shape
            if width == 1:
                x = x.reshape(batch, -1)

        return x


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_channels, out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    if x.dim() == 3:
                        batch, time, freq = x.shape
                        x = self.act(self.fc(x).reshape(batch, -1))
                    else:
                        x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            # elif x.dim() == 3:
            #     batch_size = x.shape[0]
            #     x = self.fc(x)
            #     x = self.act(x.reshape(batch_size, -1))
            else:
                x = self.act(self.fc(x))

        return x


def lstm_with_attention(net_params):
    if 'LSTM_1' in net_params:
        arguments = net_params['LSTM_1']
    else:
        arguments = net_params['GRU_1']
    if 'ATTENTION_1' in net_params and 'ATTENTION_global' not in net_params:
        if arguments[-1]:
            return 'forward'
        else:
            return 'whole'
    if 'ATTENTION_1' in net_params and 'ATTENTION_global' in net_params:
        if arguments[-1]:
            return 'forward'
        else:
            return 'whole'
    if 'ATTENTION_1' not in net_params and 'ATTENTION_global' in net_params:
        if arguments[-1]:
            return 'forward_only'
        else:
            return 'forward_only'


def reshape_x(x):
    """
    Reshapes the input 'x' if there is a dimension of length 1

    Input
        x: torch.Tensor - The input
    """
    dims = x.dim()
    if x.shape[1] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
        x = torch.reshape(x, (x.shape[0], 1))
    elif dims == 4:
        first, second, third, fourth = x.shape
        if second == 1:
            x = torch.reshape(x, (first, third, fourth))
        elif third == 1:
            x = torch.reshape(x, (first, second, fourth))
        else:
            x = torch.reshape(x, (first, second, third))
    elif dims == 3:
        first, second, third = x.shape
        if second == 1:
            x = torch.reshape(x, (first, third))
        elif third == 1:
            x = torch.reshape(x, (first, second))

    return x


class CustomAtt(nn.Module):
    """
    Use this model instance if an attention mechanism is to be used.
    Following the example of DenseNet
    (https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
    add modules to create a generator which holds the network architecture
    """
    def __init__(self, main_log, net_params):
        super(CustomAtt, self).__init__()
        d_items = list(net_params.items())
        for i, val in enumerate(net_params):
            name, number = val.split('_')
            if name == 'CONV':
                if isinstance(net_params[val][1], int):
                    layer = ConvBlock1d(in_channels=net_params[val][0][0],
                                        out_channels=net_params[val][0][1],
                                        np=net_params[val][1:],
                                        normalisation=net_params[val][-1])
                else:
                    layer = ConvBlock(in_channels=net_params[val][0][0],
                                      out_channels=net_params[val][0][1],
                                      np=net_params[val][1:],
                                      normalisation=net_params[val][-1])
            elif name == 'POOL':
                if isinstance(net_params[val][1], int):
                    layer = nn.MaxPool1d(kernel_size=net_params[val][0],
                                         stride=net_params[val][1],
                                         padding=net_params[val][2])
                else:
                    layer = nn.MaxPool2d(kernel_size=net_params[val][0],
                                         stride=net_params[val][1],
                                         padding=net_params[val][2])

            elif name == 'GLOBALPOOL':
                layer = nn.MaxPool2d(kernel_size=net_params[val][0])
            elif name == 'GRU':
                layer = nn.GRU(net_params[val][0],
                               net_params[val][1],
                               num_layers=net_params[val][2],
                               batch_first=True,
                               bidirectional=net_params[val][3])
                if net_params[val][2] > 1:
                    self.rnn_layers = True
                else:
                    self.rnn_layers = False
            elif name == 'LSTM':
                layer = nn.LSTM(net_params[val][0],
                                net_params[val][1],
                                num_layers=net_params[val][2],
                                batch_first=True,
                                bidirectional=net_params[val][3])
                if net_params[val][2] > 1:
                    self.rnn_layers = True
                else:
                    self.rnn_layers = False
            elif name == 'ATTENTION':
                att_type = net_params[val][0]
                if val == 'global':
                    activ = 'global'
                else:
                    activ = 'softmax'
                if att_type == 'conv':
                    layer = ConvBlock(in_channels=1,
                                      out_channels=1,
                                      np=net_params[val][1:4],
                                      normalisation=None,
                                      att=activ)
                else:
                    layer = FullyConnected(in_channels=net_params[val][1],
                                           out_channels=net_params[val][2],
                                           activation=activ,
                                           normalisation=None,
                                           att=True)
            elif name == 'DROP':
                layer = nn.Dropout(net_params[val])
            elif name == 'HIDDEN':
                act = self.whats_next(d_items, i+1)
                layer = FullyConnected(net_params[val][0],
                                       net_params[val][1],
                                       act,
                                       net_params[val][-1])
            elif name == 'SIGMOID' or name == 'SOFTMAX':
                break

            self.add_module(val, layer)

        main_log.info('Network Architecture:')
        main_log.info(self)

    def whats_next(self, items, value):
        """
        This is used for the instance of fully-connected layers in order to
        determine what type of activation to be used if none is specified.

        Inputs
            items: list - The items from the network parameters dictionary
            value: int - The next position in the network architecture

        Output
            next_item: str/list - Either returns sigmoid/softmax or if the
                       next layer is a fully_connected layer, the network
                       parameters
        """
        name, _ = items[value]
        next_item = name.split('_')[0]
        if next_item == 'DROP':
            return self.whats_next(items, value+1)
        elif next_item == 'HIDDEN':
            return items[1][1][0]
        elif next_item == 'SIGMOID':
            return 'sigmoid'
        elif next_item == 'SOFTMAX':
            return'softmax'
        elif next_item == 'LSTM':
            pass
        elif next_item == 'GRU':
            pass

    def init_weights(self):
        init_layer(self.fc_final)

    def forward(self, net_input, net_params, convert_to_image=False,
                hidden=None, recurrent_out=None, label='', locator=[],
                whole_train=False):
        # (samples_num, feature_maps, time_steps, freq_num)
        if convert_to_image:
            x = net_input
            data_type = '3d'
        else:
            (batch, mel_bins, seq_len) = net_input.shape
            if mel_bins == 1:
                data_type = '1d'
                x = net_input
            else:
                x = net_input.view(batch, 1, mel_bins, seq_len)
                data_type = '2d'
        z_glob = None
        prev_name = ''
        for name, layer in self.named_children():
            layer_name, layer_val = name.split('_')
            if layer_name == 'HIDDEN' and prev_name != 'HIDDEN':
                x = layer(x)
            elif layer_name == 'ATTENTION':
                if layer_val == 'global':
                    z_glob = x.clone()
                    if 'ATTENTION_1' in net_params:
                       z_glob = z_glob[:, -1, :]
                    if net_params[name][0] == 'conv':
                        if len(z_glob.shape) != 4:
                            dimensions = z_glob.shape
                            z_glob = z_glob.reshape(batch, 1, dimensions[1],
                                                    dimensions[-1])
                    z_glob = layer(z_glob)
                else:
                    z = x.clone()
                    if net_params[name][0] == 'conv':
                        if len(z.shape) != 4:
                            dimensions = z.shape
                            z = z.contiguous().view(batch, 1, dimensions[1],
                                                    dimensions[-1])
                    z = layer(z)
                    if len(z.shape) != 2:
                        z = z.reshape(batch, -1)

            elif layer_name == 'GRU' or layer_name == 'LSTM':
                ''' Get into form of Batch, Time, FeatureMap'''
                if len(x.shape) == 4:
                    if 1 in x.shape:
                        shaper = list(x.shape)
                        location = [pointer for pointer, val in enumerate(
                            shaper) if val == 1]
                        x = torch.mean(x, dim=location[0])
                    else:
                        x = torch.mean(x, dim=2)
                x = x.transpose(1, 2)
                if layer_name == 'LSTM':
                    if hidden is None:
                        x, (h, c) = layer(x)
                    else:
                        x, (h, c) = layer(x, hidden)
                else:
                    if hidden is None:
                        x, h = layer(x)
                    else:
                        x, h = layer(x, hidden)

                if 'ATTENTION_1' in net_params or 'ATTENTION_global' in \
                        net_params:
                    recurrent_out = lstm_with_attention(net_params)
                arguments = net_params[name]
                # If we are using BiDi LSTM/GRU
                if arguments[-1]:
                    if 'LSTM_2' in net_params or 'GRU_2' in net_params:
                        x = x
                    else:
                        if recurrent_out == 'whole':
                            x = x
                        elif recurrent_out == 'forward':
                            dim = x.shape[-1]
                            dim = (dim // 2)
                            x = x[:, :, 0:dim]
                        elif recurrent_out == 'forward_only':
                            x = h[-2]
                        elif recurrent_out == 'forward_backward':
                            x = h
                            x = x.transpose(0, 1)
                            x = x.reshape(batch, -1)
                else:
                    if 'LSTM_2' in net_params or 'GRU_2' in net_params:
                        x = x
                    else:
                        if recurrent_out == 'whole':
                            x = x
                        elif recurrent_out == 'forward_only':
                            x = h[-1]
                if 'ATTENTION_1' in net_params:
                    att_type = net_params['ATTENTION_1'][-1]
                    if att_type == 'feature':
                        x = x.transpose(1, 2)
            elif layer_name == 'POOL':
                if x.dim() != 3 and data_type == '1d':
                    x = x.contiguous().view(batch, 1, -1)
                x = layer(x)
            else:
                x = layer(x)
            prev_name = layer_name

        if 'ATTENTION_1' in net_params:
            x = x.reshape(batch, -1)
            x = x * z
            x = torch.sum(x, dim=1).reshape(batch, -1)

        if 'GRU_1' in net_params:
            hc = h
            if x.dim() == 3:
                x = torch.mean(x, dim=1)
            if 'HIDDEN_1' not in net_params:
                if 'SOFTMAX_1' in net_params:
                    s = nn.Softmax(dim=-1)
                elif 'SIGMOID_1' in net_params:
                    s = nn.Sigmoid()
                x = s(x)
            return x, hc, z_glob
        elif 'LSTM_1' in net_params:
            hc = (h, c)
            if x.dim() == 3:
                x = torch.mean(x, dim=1)
            if 'HIDDEN_1' not in net_params:
                if 'SOFTMAX_1' in net_params:
                    s = nn.Softmax(dim=-1)
                elif 'SIGMOID_1' in net_params:
                    s = nn.Sigmoid()
                x = s(x)
            return x, hc, z_glob
        else:
            if 'HIDDEN_1' not in net_params:
                if 'SOFTMAX_1' in net_params:
                    s = nn.Softmax(dim=-1)
                elif 'SIGMOID_1' in net_params:
                    s = nn.Sigmoid()
                x = s(x)
            return x, None, z_glob


class Custom(nn.Module):
    def __init__(self, main_log, net_params):
        super(Custom, self).__init__()
        d_items = list(net_params.items())
        for i, val in enumerate(net_params):
            name, number = val.split('_')
            if name == 'CONV':
                if isinstance(net_params[val][1], int):
                    layer = ConvBlock1d(in_channels=net_params[val][0][0],
                                        out_channels=net_params[val][0][1],
                                        np=net_params[val][1:],
                                        normalisation=net_params[val][-1])
                else:
                    layer = ConvBlock(in_channels=net_params[val][0][0],
                                      out_channels=net_params[val][0][1],
                                      np=net_params[val][1:],
                                      normalisation=net_params[val][-1])
            elif name == 'POOL':
                if isinstance(net_params[val][1], int):
                    layer = nn.MaxPool1d(kernel_size=net_params[val][0],
                                         stride=net_params[val][1],
                                         padding=net_params[val][2])
                else:
                    layer = nn.MaxPool2d(kernel_size=net_params[val][0],
                                         stride=net_params[val][1],
                                         padding=net_params[val][2])

            elif name == 'GLOBALPOOL':
                layer = nn.MaxPool2d(kernel_size=net_params[val][0])
            elif name == 'GRU':
                layer = nn.GRU(net_params[val][0],
                               net_params[val][1],
                               num_layers=net_params[val][2],
                               batch_first=True,
                               dropout=.9,
                               bidirectional=net_params[val][3])
                if net_params[val][2] > 1:
                    self.rnn_layers = True
                else:
                    self.rnn_layers = False
            elif name == 'LSTM':
                layer = nn.LSTM(net_params[val][0],
                                net_params[val][1],
                                num_layers=net_params[val][2],
                                batch_first=True,
                                bidirectional=net_params[val][3])
                if net_params[val][2] > 1:
                    self.rnn_layers = True
                else:
                    self.rnn_layers = False
            elif name == 'DROP':
                layer = nn.Dropout(net_params[val])
            elif name == 'HIDDEN':
                act = self.whats_next(d_items, i + 1)
                layer = FullyConnected(net_params[val][0],
                                       net_params[val][1],
                                       act,
                                       net_params[val][-1])
            elif name == 'SIGMOID' or name == 'SOFTMAX':
                break

            self.add_module(val, layer)

        main_log.info('Network Architecture:')
        main_log.info(self)

    def whats_next(self, items, value):
        name, _ = items[value]
        next_item = name.split('_')[0]
        if next_item == 'DROP':
            return self.whats_next(items, value + 1)
        elif next_item == 'HIDDEN':
            pass
        elif next_item == 'SIGMOID':
            return 'sigmoid'
        elif next_item == 'SOFTMAX':
            return 'softmax'
        elif next_item == 'LSTM':
            pass
        elif next_item == 'GRU':
            pass

    def init_weights(self):
        init_layer(self.fc_final)

    def reshape_x(self, x):
        dims = x.dim()
        if x.shape[1] == 1 and x.shape[2] == 1 and x.shape[3] == 1:
            x = torch.reshape(x, (x.shape[0], 1))
        elif dims == 4:
            first, second, third, fourth = x.shape
            if second == 1:
                x = torch.reshape(x, (first, third, fourth))
            elif third == 1:
                x = torch.reshape(x, (first, second, fourth))
            else:
                x = torch.reshape(x, (first, second, third))
        elif dims == 3:
            first, second, third = x.shape
            if second == 1:
                x = torch.reshape(x, (first, third))
            elif third == 1:
                x = torch.reshape(x, (first, second))

        return x

    def forward(self, net_input, net_params, convert_to_image=False,
                hidden=None, recurrent_out=None, label='', locator=[],
                whole_train=False):
        # (samples_num, feature_maps, time_steps, freq_num)
        if convert_to_image:
            x = net_input
            data_type = '3d'
        else:
            (batch, mel_bins, seq_len) = net_input.shape
            if mel_bins == 1:
                data_type = '1d'
                x = net_input
            else:
                x = net_input.view(batch, 1, mel_bins, seq_len)
                data_type = '2d'
        prev_name = ''
        for name, layer in self.named_children():
            layer_name = name.split('_')[0]
            if layer_name == 'HIDDEN' and prev_name != 'HIDDEN':
                index = None
                if x.dim() == 3:
                    x = x.transpose(1, 2)
                    x = layer(x)
                    for place, number_d in enumerate(x.size()):
                        first, second, third = x.shape
                        if number_d == 1:
                            index = place
                    if index:
                        if index == 1:
                            x = torch.reshape(x, (first, third))
                        elif index == 2:
                            x = torch.reshape(x, (first, second))
                else:
                    x = layer(x)
            elif layer_name == 'GRU' or layer_name == 'LSTM':
                ''' Get into form of Batch, Time, FeatureMap'''
                if len(x.shape) == 4:
                    if 1 in x.shape:
                        shaper = list(x.shape)
                        location = [pointer for pointer, val in enumerate(
                            shaper) if val == 1 and pointer > 0]
                        x = torch.mean(x, dim=location[0])
                    else:
                        x = torch.mean(x, dim=2)
                x = x.transpose(1, 2)

                if layer_name == 'LSTM':
                    if hidden is None:
                        x, (h, c) = layer(x)
                    else:
                        x, (h, c) = layer(x, hidden)
                else:
                    if hidden is None:
                        x, h = layer(x)
                    else:
                        x, h = layer(x, hidden)
                arguments = net_params[name]
                # If we are using BiDi LSTM/GRU
                if arguments[-1]:
                    if 'LSTM_2' in net_params or 'GRU_2' in net_params:
                        x = x
                    else:
                        if recurrent_out == 'whole':
                            x = x
                        elif recurrent_out == 'forward':
                            dim = x.shape[-1]
                            dim = (dim // 2)
                            x = x[:, :, 0:dim]
                        elif recurrent_out == 'forward_only':
                            x = h[-2]
                        elif recurrent_out == 'forward_backward':
                            x = h
                            x = x.transpose(0, 1)
                            x = x.contiguous().view(batch, -1)
                else:
                    if 'LSTM_2' in net_params or 'GRU_2' in net_params:
                        x = x
                    else:
                        if recurrent_out == 'whole':
                            x = x
                        elif recurrent_out == 'forward_only':
                            x = h[-1]
            elif name.split('_')[0] == 'POOL':
                if x.dim() != 3 and data_type == '1d':
                    x = x.contiguous().view(batch, 1, -1)
                x = layer(x)
            else:
                x = layer(x)

            prev_name = layer_name

        if 'GRU_1' in net_params:
            hc = h
            if x.dim() == 3:
                x = torch.mean(x, dim=1)
            if 'HIDDEN_1' not in net_params:
                if 'SOFTMAX_1' in net_params:
                    s = nn.Softmax(dim=-1)
                elif 'SIGMOID_1' in net_params:
                    s = nn.Sigmoid()
                x = s(x)
            return x, hc, None
        elif 'LSTM_1' in net_params:
            hc = (h, c)
            if x.dim() == 3:
                x = torch.mean(x, dim=1)
            if 'HIDDEN_1' not in net_params:
                if 'SOFTMAX_1' in net_params:
                    s = nn.Softmax(dim=-1)
                elif 'SIGMOID_1' in net_params:
                    s = nn.Sigmoid()
                x = s(x)
            return x, hc, None
        else:
            if 'HIDDEN_1' not in net_params:
                if 'SOFTMAX_1' in net_params:
                    s = nn.Softmax(dim=-1)
                elif 'SIGMOID_1' in net_params:
                    s = nn.Sigmoid()
                x = s(x)
            return x, None, None

import torch
import torch.nn as nn


def create_tensor_data(x, cuda, select_net, labels=False):
    """
    Converts the data from numpy arrays to torch tensors

    Inputs
        x: The input data
        cuda: Bool - Set to true if using the GPU
        select_net: Slightly different processing if using multi-gpu option

    Output
        x: Data converted to a tensor
    """
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        raise Exception("Error!")

    if not labels:
        if cuda and select_net == 'custom_multigpu':
            x = x.cuda(0)
        elif cuda:
            x = x.cuda()

    return x


def calculate_loss(prediction, target, cw=None, net_params='SOFTMAX_1',
                   gender=True):
    """
    With respect to the final layer of the model, calculate the loss of the
    model.

    Inputs
        prediction: The output of the model
        target: The relative label for the output of the model
        averaging_setting: Geometric or arithmetic
        cw: The class weights for the dataset
        sigmoid_processing: round or unnorm. Round treats .5 as the threshold
                            whereas unnorm is the unnormalised probability.
                            In this case, everything is relative to class 0.
                            E.g. output of 0.8 and label=0 is 80% probability
                            of class 0, whereas 0.8 and label=1 is 20%
                            probability of class 1
        net_params: Dictionary containing the model configuration

    Output
        loss: The BCELoss or NLL_Loss
    """
    if 'SIGMOID_1' in net_params:
        loss_func = 'SIG'
    else:
        loss_func = 'SFMX'

    if loss_func == 'SFMX':
        loss = torch.nn.functional.nll_loss(prediction,
                                            target,
                                            weight=cw)
    elif loss_func == 'SIG':
        # batch_zeros_f, batch_ones_f, batch_zeros_m, batch_ones_m
        # cw = f_ndep_ind, f_dep_ind, m_ndep_ind, m_dep_ind
        # OR
        # cw = nd, d
        if gender:
            fem_nd_w, fem_d_w, male_nd_w, male_d_w = cw
            zero_ind = (target == 0).nonzero().reshape(-1)
            one_ind = (target == 1).nonzero().reshape(-1)
            two_ind = (target == 2).nonzero().reshape(-1)
            three_ind = (target == 3).nonzero().reshape(-1)
            class_weights = torch.ones(target.shape[0])
            class_weights.scatter_(0, zero_ind, fem_nd_w[0])
            class_weights.scatter_(0, one_ind, fem_d_w[0])
            class_weights.scatter_(0, two_ind, male_nd_w[0])
            class_weights.scatter_(0, three_ind, male_d_w[0])
            cw = class_weights.reshape(-1, 1)
            target = target % 2
        else:
            if target.shape[0] != cw.shape[0]:
                zero_ind = (target == 0).nonzero().reshape(-1)
                one_ind = (target == 1).nonzero().reshape(-1)
                class_weights = torch.ones(target.shape[0])
                class_weights.scatter_(0, zero_ind, cw[0])
                class_weights.scatter_(0, one_ind, cw[1])

                cw = class_weights.reshape(-1, 1)
        if type(cw) is not torch.Tensor:
            cw = torch.Tensor(cw)
        if net_params['SIGMOID_1'] == 'unnorm':
            target = 1 - target
        if prediction.dim() == 1:
            prediction = prediction.view(-1, 1)
        bceloss = nn.BCELoss(weight=cw)
        loss = bceloss(prediction, target.float().view(-1, 1))

    return loss


def bce_loss_func(target, prediction, avg_setting='arithmetic', weight=None):
    """
    Calculates the Binary Cross Entropy Loss. In the simplest case, it takes
    the targets and predictions and calculates l = (1-y)log(1-x) + ylog(x).
    If the geometric averaging system is used then the NLL_Loss will be used
    as the data must be presented in the form of p(class_0), p(class_1) even
    if the final layer is sigmoid. This is because geometric averaging means
    taking the log before averaging which will mean that for the case of
    class 0, the incorrect values will be used to calculate the loss. Weights
    can also be applied.

    Inputs:
        target: The labels of the data
        prediction: The output of the network
        avg_setting: This can either be arithmetic or geometric
        weight: If provided will apply a weight in the case of imbalanced data

    Output:
        loss: The BCE loss
    """
    if avg_setting == 'arithmetic':
        out_zeros = (1 - target) * torch.log(1 - prediction)
        out_ones = target * torch.log(prediction)
        if weight is not None:
            if type(weight) is not torch.Tensor:
                weight = torch.Tensor(weight)
            out_zeros = out_zeros * weight[0]
            out_ones = out_ones * weight[1]

        loss = -torch.mean(out_zeros + out_ones)
    else:
        loss = torch.nn.functional.nll_loss(prediction,
                                            target,
                                            weight)
    return loss

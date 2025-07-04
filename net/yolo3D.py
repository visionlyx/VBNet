import torch
import torch.nn as nn
from collections import OrderedDict
from net.darknet import darknet_vessel



def conv3d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv3d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm3d(filter_out)),
        ("relu", nn.LeakyReLU(0.3)),
    ]))


def make_last_layers(filters_list, in_filters, out_filter):
    m = nn.ModuleList([
        conv3d(in_filters, filters_list[0], 1),
        conv3d(filters_list[0], filters_list[1], 3),
        conv3d(filters_list[1], filters_list[0], 1),
        conv3d(filters_list[0], filters_list[1], 3),
        conv3d(filters_list[1], filters_list[0], 1),
        conv3d(filters_list[0], filters_list[1], 3),
        nn.Conv3d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m


class Yolo3DBody(nn.Module):
    def __init__(self, config):
        super(Yolo3DBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet_vessel()

        out_filters = self.backbone.layers_out_filters
        #  last_layer0
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.last_layer0 = make_last_layers([256, 512], out_filters[-1], final_out_filter0)

        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv3d(256, 128, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([128, 256], out_filters[-2] + 128, final_out_filter1)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch
        #  backbone
        x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0, out0_branch = _branch(self.last_layer0, x0)

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)


        return out0, out1


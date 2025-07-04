import torch
import torch.nn as nn
from collections import OrderedDict
#from net.darknet import darknet_vessel
from net.darknet53 import darknet_vessel



def conv3d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv3d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm3d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
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

def make_last_layers_no_out_put(filters_list, in_filters):
    m = nn.ModuleList([
        conv3d(in_filters, filters_list[0], 1),
        conv3d(filters_list[0], filters_list[1], 3),
        conv3d(filters_list[1], filters_list[0], 1),
        conv3d(filters_list[0], filters_list[1], 3),
        conv3d(filters_list[1], filters_list[0], 1),

    ])
    return m





class Yolo3DBody(nn.Module):
    def __init__(self, config):
        super(Yolo3DBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet_vessel()

        out_filters = self.backbone.layers_out_filters
        #  last_layer0  no output
        self.last_layer0 = make_last_layers_no_out_put([512, 1024], out_filters[-1])

        #  embedding1
        final_out_filter1 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv3d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1)

        final_out_filter2 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv3d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2)


    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
            return layer_in, out_branch


        def _branch_last(last_layer, layer_in):
            for i, e in enumerate(last_layer):
                layer_in = e(layer_in)
            return layer_in


        #  backbone
        x2, x1, x0 = self.backbone(x)
        #  yolo branch 0
        out0_branch = _branch_last(self.last_layer0, x0)

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out1, out1_branch = _branch(self.last_layer1, x1_in)


        x2_in = self.last_layer2_conv(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out2, _ = _branch(self.last_layer2, x2_in)


        return out1, out2


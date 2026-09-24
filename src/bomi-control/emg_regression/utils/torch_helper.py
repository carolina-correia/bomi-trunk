#!/usr/bin/env python

import os
import torch


class TorchHelper():
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), os.path.join('', '{}.pt'.format(path)))

    @staticmethod
    def load(model, path, device):
        model.load_state_dict(torch.load(os.path.join('', '{}.pt'.format(path)), map_location=torch.device(device)))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:21:46 2022

@author: yoann
"""

from torch_geometric.transforms import BaseTransform
class NormalizeFingerprint(BaseTransform):
    r"""Normalizes node features to the interval :math:`(-1, 1)`.
    """
    def __call__(self, data):
        range_ = 225-30
        normalized = ((data.x-30)/range_)*2 - 1
        data.x = normalized
        return data
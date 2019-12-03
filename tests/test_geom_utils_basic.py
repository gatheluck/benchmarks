import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import math
import torch

from modules.geom_utils import get_quat_inplane

def test_get_quat_inplane():
    B = 16
    eye = torch.FloatTensor([0,0,-2.732]).unsqueeze(0).repeat(B,1)
    radian = torch.randn(B,1) * math.pi/3.0
    quat = get_quat_inplane(eye, radian)
    
    assert len(quat.size()) == 2
    assert quat.size(0) == B
    assert quat.size(1) == 4
    for i in range(B):
        assert abs(torch.norm(quat[i,:]).item()-1.0) < 0.000001
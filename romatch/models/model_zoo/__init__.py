from typing import Union
import torch
from .roma_models import roma_model

weight_urls = "https://huggingface.co/datasets/Liangyingping/L2Mpp-checkpoints/resolve/main/l2m%2B%2B_560x560.pth"
weight_urlsv2 = "https://huggingface.co/datasets/Liangyingping/L2Mpp-checkpoints/resolve/main/l2m%2B%2B_560x560v2.pth.ckpt"

def l2mpp_model(device, weights=None, dinov2_weights=None, coarse_res = 672, upsample_res = 1344, amp_dtype: torch.dtype = torch.float16, version="v1"):
    if isinstance(coarse_res, int):
        coarse_res = (coarse_res, coarse_res)
    if isinstance(upsample_res, int):    
        upsample_res = (upsample_res, upsample_res)

    if str(device) == 'cpu':
        amp_dtype = torch.float32

    assert coarse_res[0] % 14 == 0, "Needs to be multiple of 14 for backbone"
    assert coarse_res[1] % 14 == 0, "Needs to be multiple of 14 for backbone"
    
    if weights is None:
        if version == "v1":
            weight_urls2load = weight_urls
        elif version == "v2":
            weight_urls2load = weight_urlsv2
        else:
            raise NotImplementedError

        print("[loading]:", weight_urls)
        weights = torch.hub.load_state_dict_from_url(weight_urls, map_location=device)
    model = roma_model(resolution=coarse_res, upsample_preds=True,
               weights=weights,dinov2_weights = dinov2_weights,device=device, amp_dtype=amp_dtype)
    model.upsample_res = upsample_res
    print(f"Using coarse resolution {coarse_res}, and upsample res {model.upsample_res}")
    return model

from romatch import l2mpp_model
import torch
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

l2mpp = l2mpp_model(device=device, version="v2") # , version="v1"
# Match
warp, certainty = l2mpp.match("assets/sacre_coeur_A.jpg", "assets/sacre_coeur_B.jpg", device=device)
matches, certainty = l2mpp.sample(warp, certainty)

# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
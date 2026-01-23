from romatch import l2mpp_model
import torch
import cv2

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

l2mpp = l2mpp_model(device=device)
# Match
warp, certainty = l2mpp.match("assets/sacre_coeur_A.jpg", "assets/sacre_coeur_A.jpg", device=device)
matches, certainty = l2mpp.sample(warp, certainty)
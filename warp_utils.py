import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes
import numpy as np
from functools import reduce

def vis_depth_discontinuity(
    depth, depth_threshold, vis_diff=False, label=False, mask=None
):
    if label == False:
        disp = 1.0 / depth
        u_diff = (disp[1:, :] - disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] - disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] - disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] - disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > depth_threshold).astype(np.float32)
        b_over = (np.abs(b_diff) > depth_threshold).astype(np.float32)
        l_over = (np.abs(l_diff) > depth_threshold).astype(np.float32)
        r_over = (np.abs(r_diff) > depth_threshold).astype(np.float32)
    else:
        disp = depth
        u_diff = (disp[1:, :] * disp[:-1, :])[:-1, 1:-1]
        b_diff = (disp[:-1, :] * disp[1:, :])[1:, 1:-1]
        l_diff = (disp[:, 1:] * disp[:, :-1])[1:-1, :-1]
        r_diff = (disp[:, :-1] * disp[:, 1:])[1:-1, 1:]
        if mask is not None:
            u_mask = (mask[1:, :] * mask[:-1, :])[:-1, 1:-1]
            b_mask = (mask[:-1, :] * mask[1:, :])[1:, 1:-1]
            l_mask = (mask[:, 1:] * mask[:, :-1])[1:-1, :-1]
            r_mask = (mask[:, :-1] * mask[:, 1:])[1:-1, 1:]
            u_diff = u_diff * u_mask
            b_diff = b_diff * b_mask
            l_diff = l_diff * l_mask
            r_diff = r_diff * r_mask
        u_over = (np.abs(u_diff) > 0).astype(np.float32)
        b_over = (np.abs(b_diff) > 0).astype(np.float32)
        l_over = (np.abs(l_diff) > 0).astype(np.float32)
        r_over = (np.abs(r_diff) > 0).astype(np.float32)
    u_over = np.pad(u_over, 1, mode="constant")
    b_over = np.pad(b_over, 1, mode="constant")
    l_over = np.pad(l_over, 1, mode="constant")
    r_over = np.pad(r_over, 1, mode="constant")
    u_diff = np.pad(u_diff, 1, mode="constant")
    b_diff = np.pad(b_diff, 1, mode="constant")
    l_diff = np.pad(l_diff, 1, mode="constant")
    r_diff = np.pad(r_diff, 1, mode="constant")

    if vis_diff:
        return [u_over, b_over, l_over, r_over], [u_diff, b_diff, l_diff, r_diff]
    else:
        return [u_over, b_over, l_over, r_over]

def rolling_window(a, window, strides):
    assert (
        len(a.shape) == len(window) == len(strides)
    ), "'a', 'window', 'strides' dimension mismatch"
    shape_fn = lambda i, w, s: (a.shape[i] - w) // s + 1
    shape = [shape_fn(i, w, s) for i, (w, s) in enumerate(zip(window, strides))] + list(
        window
    )

    def acc_shape(i):
        if i + 1 >= len(a.shape):
            return 1
        else:
            return reduce(lambda x, y: x * y, a.shape[i + 1 :])

    _strides = [acc_shape(i) * s * a.itemsize for i, s in enumerate(strides)] + list(
        a.strides
    )

    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=_strides)

def bilateral_filter(
    depth,
    sigma_s,
    sigma_r,
    window_size,
    discontinuity_map=None,
    HR=False,
    mask=None,
):

    midpt = window_size // 2
    ax = np.arange(-midpt, midpt + 1.0)
    xx, yy = np.meshgrid(ax, ax)
    if discontinuity_map is not None:
        spatial_term = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma_s ** 2))

    # padding
    depth = depth[1:-1, 1:-1]
    depth = np.pad(depth, ((1, 1), (1, 1)), "edge")
    pad_depth = np.pad(depth, (midpt, midpt), "edge")
    if discontinuity_map is not None:
        discontinuity_map = discontinuity_map[1:-1, 1:-1]
        discontinuity_map = np.pad(discontinuity_map, ((1, 1), (1, 1)), "edge")
        pad_discontinuity_map = np.pad(discontinuity_map, (midpt, midpt), "edge")
        pad_discontinuity_hole = 1 - pad_discontinuity_map
    # filtering
    output = depth.copy()
    pad_depth_patches = rolling_window(pad_depth, [window_size, window_size], [1, 1])
    if discontinuity_map is not None:
        pad_discontinuity_patches = rolling_window(
            pad_discontinuity_map, [window_size, window_size], [1, 1]
        )
        pad_discontinuity_hole_patches = rolling_window(
            pad_discontinuity_hole, [window_size, window_size], [1, 1]
        )

    if mask is not None:
        pad_mask = np.pad(mask, (midpt, midpt), "constant")
        pad_mask_patches = rolling_window(pad_mask, [window_size, window_size], [1, 1])
    from itertools import product

    if discontinuity_map is not None:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if mask is not None and mask[pi, pj] == 0:
                    continue
                if discontinuity_map is not None:
                    if bool(pad_discontinuity_patches[pi, pj].any()) is False:
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = pad_discontinuity_hole_patches[pi, pj]
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size // 2, window_size // 2]
                if discontinuity_map is not None:
                    coef = discontinuity_holes.astype(np.float32)
                    if mask is not None:
                        coef = coef * pad_mask_patches[pi, pj]
                else:
                    range_term = np.exp(
                        -((depth_patch - patch_midpt) ** 2) / (2.0 * sigma_r ** 2)
                    )
                    coef = spatial_term * range_term
                if coef.max() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.max() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef / (coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]
    else:
        pH, pW = pad_depth_patches.shape[:2]
        for pi in range(pH):
            for pj in range(pW):
                if discontinuity_map is not None:
                    if (
                        pad_discontinuity_patches[pi, pj][
                            window_size // 2, window_size // 2
                        ]
                        == 1
                    ):
                        continue
                    discontinuity_patch = pad_discontinuity_patches[pi, pj]
                    discontinuity_holes = 1.0 - discontinuity_patch
                depth_patch = pad_depth_patches[pi, pj]
                depth_order = depth_patch.ravel().argsort()
                patch_midpt = depth_patch[window_size // 2, window_size // 2]
                range_term = np.exp(
                    -((depth_patch - patch_midpt) ** 2) / (2.0 * sigma_r ** 2)
                )
                if discontinuity_map is not None:
                    coef = spatial_term * range_term * discontinuity_holes
                else:
                    coef = spatial_term * range_term
                if coef.sum() == 0:
                    output[pi, pj] = patch_midpt
                    continue
                if discontinuity_map is not None and (coef.sum() == 0):
                    output[pi, pj] = patch_midpt
                else:
                    coef = coef / (coef.sum())
                    coef_order = coef.ravel()[depth_order]
                    cum_coef = np.cumsum(coef_order)
                    ind = np.digitize(0.5, cum_coef)
                    output[pi, pj] = depth_patch.ravel()[depth_order][ind]

    return output

class RGBDRenderer:
    def __init__(self, device):
        self.device = device
        self.eps = 0.1
        self.near_z = 1e-2
        self.far_z = 1e4
    
    def render_mesh(self, mesh_dict, cam_int, cam_ext):
        vertice = mesh_dict["vertice"]  # [b,h*w,3]
        faces = mesh_dict["faces"]  # [b,nface,3]
        attributes = mesh_dict["attributes"]  # [b,h*w,4]
        h, w = mesh_dict["size"]

        ############
        # to NDC space
        vertice_homo = self.lift_to_homo(vertice)  # [b,h*w,4]
        # [b,1,3,4] x [b,h*w,4,1] = [b,h*w,3,1]
        vertice_world = torch.matmul(cam_ext.unsqueeze(1), vertice_homo[..., None]).squeeze(-1)  # [b,h*w,3]
        vertice_depth = vertice_world[..., -1:]  # [b,h*w,1]
        attributes = torch.cat([attributes, vertice_depth], dim=-1)  # [b,h*w,5]
        # [b,1,3,3] x [b,h*w,3,1] = [b,h*w,3,1]
        vertice_world_homo = self.lift_to_homo(vertice_world)
        persp = self.get_perspective_from_intrinsic(cam_int)  # [b,4,4]

        # [b,1,4,4] x [b,h*w,4,1] = [b,h*w,4,1]
        vertice_ndc = torch.matmul(persp.unsqueeze(1), vertice_world_homo[..., None]).squeeze(-1)  # [b,h*w,4]
        vertice_ndc = vertice_ndc[..., :-1] / vertice_ndc[..., -1:]
        vertice_ndc[..., :-1] *= -1
        vertice_ndc[..., 0] *= w / h

        ############
        # render
        mesh = Meshes(vertice_ndc, faces)
        pix_to_face, _, bary_coords, _ = rasterize_meshes(mesh, (h, w), faces_per_pixel=1, blur_radius=1e-6)  # [b,h,w,1] [b,h,w,1,3]

        b, nf, _ = faces.size()
        faces = faces.reshape(b, nf * 3, 1).repeat(1, 1, 6)  # [b,3f,5]
        face_attributes = torch.gather(attributes, dim=1, index=faces)  # [b,3f,5]
        face_attributes = face_attributes.reshape(b * nf, 3, 6)
        output = interpolate_face_attributes(pix_to_face, bary_coords, face_attributes)
        output = output.squeeze(-2).permute(0, 3, 1, 2)

        render = output[:, :3]
        mask = output[:, 3:4]
        object_mask = output[:, 4:5]
        disparity = torch.reciprocal(output[:, 5:] + 1e-4)

        return render * mask, disparity * mask, mask, object_mask

    def construct_mesh(self, rgbd, cam_int, obj_mask, normalize_depth=False):
        b, _, h, w = rgbd.size()
        
        ############
        # get pixel coordinates
        pixel_2d = self.get_screen_pixel_coord(h, w)  # [1,h,w,2]
        pixel_2d_homo = self.lift_to_homo(pixel_2d)  # [1,h,w,3]

        ############
        # project pixels to 3D space
        rgbd = rgbd.permute(0, 2, 3, 1)  # [b,h,w,4]
        disparity = rgbd[..., -1:]  # [b,h,w,1]
        depth = torch.reciprocal(disparity +  + 1e-4) # [b,h,w,1]
        obj_mask = obj_mask.permute(0, 2, 3, 1).to(depth.device)
        # In [2]: depth.max()
        # Out[2]: 3.0927802771530017

        # In [3]: depth.min()
        # Out[3]: 1.466965406649775
        cam_int_inv = torch.inverse(cam_int)  # [b,3,3]
        # [b,1,1,3,3] x [1,h,w,3,1] = [b,h,w,3,1]
        pixel_3d = torch.matmul(cam_int_inv[:, None, None, :, :], pixel_2d_homo[..., None]).squeeze(-1)  # [b,h,w,3]

        pixel_3d = pixel_3d * depth  # [b,h,w,3]
        vertice = pixel_3d.reshape(b, h * w, 3)  # [b,h*w,3]
        ############
        # construct faces
        faces = self.get_faces(h, w)  # [1,nface,3]
        faces = faces.repeat(b, 1, 1).long()  # [b,nface,3]

        ############
        # compute attributes
        attr_color = rgbd[..., :-1].reshape(b, h * w, 3)  # [b,h*w,3]
        attr_object = obj_mask.reshape(b, h * w, 1).to(attr_color.device)  # [b,h*w,1]
        attr_mask = self.get_visible_mask(disparity, alpha_threshold=0.1).reshape(b, h * w, 1)  # [b,h*w,1]
        attr = torch.cat([attr_color, attr_mask, attr_object], dim=-1)  # [b,h*w,4]
        mesh_dict = {
            "vertice": vertice,
            "faces": faces,
            "attributes": attr,
            "size": [h, w],
        }
        return mesh_dict

    def get_screen_pixel_coord(self, h, w):
        '''
        get normalized pixel coordinates on the screen
        x to left, y to down
        
        e.g.
        [0,0][1,0][2,0]
        [0,1][1,1][2,1]
        output:
            pixel_coord: [1,h,w,2]
        '''
        x = torch.arange(w).to(self.device)  # [w]
        y = torch.arange(h).to(self.device)  # [h]
        x = (x + 0.5) / w
        y = (y + 0.5) / h
        x = x[None, None, ..., None].repeat(1, h, 1, 1)  # [1,h,w,1]
        y = y[None, ..., None, None].repeat(1, 1, w, 1)  # [1,h,w,1]
        pixel_coord = torch.cat([x, y], dim=-1)  # [1,h,w,2]
        return pixel_coord
    
    def lift_to_homo(self, coord):
        '''
        return the homo version of coord
        input: coord [..., k]
        output: homo_coord [...,k+1]
        '''
        ones = torch.ones_like(coord[..., -1:])
        return torch.cat([coord, ones], dim=-1)

    def get_faces(self, h, w):
        x = torch.arange(w - 1).to(self.device)  # [w-1]
        y = torch.arange(h - 1).to(self.device)  # [h-1]
        x = x[None, None, ..., None].repeat(1, h - 1, 1, 1)  # [1,h-1,w-1,1]
        y = y[None, ..., None, None].repeat(1, 1, w - 1, 1)  # [1,h-1,w-1,1]

        tl = y * w + x
        tr = y * w + x + 1
        bl = (y + 1) * w + x
        br = (y + 1) * w + x + 1

        faces_l = torch.cat([tl, bl, br], dim=-1).reshape(1, -1, 3)  # [1,(h-1)(w-1),3]
        faces_r = torch.cat([br, tr, tl], dim=-1).reshape(1, -1, 3)  # [1,(h-1)(w-1),3]

        return torch.cat([faces_l, faces_r], dim=1)  # [1,nface,3]

    def get_visible_mask(self, disparity, beta=10, alpha_threshold=0.3):
        b, h, w, _ = disparity.size()
        disparity = disparity.reshape(b, 1, h, w)  # [b,1,h,w]
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).float().to(self.device)
        sobel_x = F.conv2d(disparity, kernel_x, padding=(1, 1))  # [b,1,h,w]
        sobel_y = F.conv2d(disparity, kernel_y, padding=(1, 1))  # [b,1,h,w]
        sobel_mag = torch.sqrt(sobel_x ** 2 + sobel_y ** 2).reshape(b, h, w, 1)  # [b,h,w,1]
        alpha = torch.exp(-1.0 * beta * sobel_mag)  # [b,h,w,1]
        vis_mask = torch.greater(alpha, alpha_threshold).float()
        return vis_mask

    def get_perspective_from_intrinsic(self, cam_int):
        '''
        input:
            cam_int: [b,3,3]
        
        output:
            persp: [b,4,4]
        '''
        fx, fy = cam_int[:, 0, 0], cam_int[:, 1, 1]  # [b]
        cx, cy = cam_int[:, 0, 2], cam_int[:, 1, 2]  # [b]

        one = torch.ones_like(cx)  # [b]
        zero = torch.zeros_like(cx)  # [b]

        near_z, far_z = self.near_z * one, self.far_z * one
        a = (near_z + far_z) / (far_z - near_z)
        b = -2.0 * near_z * far_z / (far_z - near_z)

        matrix = [[2.0 * fx, zero, 2.0 * cx - 1.0, zero],
                  [zero, 2.0 * fy, 2.0 * cy - 1.0, zero],
                  [zero, zero, a, b],
                  [zero, zero, one, zero]]
        # -> [[b,4],[b,4],[b,4],[b,4]] -> [b,4,4]        
        persp = torch.stack([torch.stack(row, dim=-1) for row in matrix], dim=-2)  # [b,4,4]
        # print(fx, cx, cy, a, b)
        return persp


#######################
# some helper I/O functions
#######################
def image_to_tensor(img_path, unsqueeze=True):
    rgb = transforms.ToTensor()(Image.open(img_path))
    if unsqueeze:
        rgb = rgb.unsqueeze(0)
    return rgb

def sparse_bilateral_filtering(
    depth,
    filter_size,
    sigma_r=0.5,
    sigma_s=4.0,
    depth_threshold=0.04,
    HR=False,
    mask=None,
    num_iter=None,
):

    save_discontinuities = []
    vis_depth = depth.copy()
    for i in range(num_iter):
        u_over, b_over, l_over, r_over = vis_depth_discontinuity(
            vis_depth, depth_threshold, mask=mask
        )

        discontinuity_map = (u_over + b_over + l_over + r_over).clip(0.0, 1.0)
        discontinuity_map[depth == 0] = 1
        save_discontinuities.append(discontinuity_map)
        if mask is not None:
            discontinuity_map[mask == 0] = 0
        vis_depth = bilateral_filter(
            vis_depth,
            sigma_r=sigma_r,
            sigma_s=sigma_s,
            discontinuity_map=discontinuity_map,
            HR=HR,
            mask=mask,
            window_size=filter_size[i],
        )

    return vis_depth

def disparity_to_tensor(disp_path, unsqueeze=True):
    disp = cv2.imread(disp_path, -1) / (2 ** 16 - 1)
    disp = sparse_bilateral_filtering(disp + 1e-4, filter_size=[5, 5], num_iter=2)
    disp = torch.from_numpy(disp)[None, ...]
    if unsqueeze:
        disp = disp.unsqueeze(0)
    return disp.float()


#######################
# some helper geometry functions
# adapt from https://github.com/mattpoggi/depthstillation
#######################
def transformation_from_parameters(axisangle, translation, invert=False):
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous().view(-1, 3, 1)
    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t
    return T


def rot_from_axisangle(vec):
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


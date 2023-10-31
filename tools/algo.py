import torch
import numpy as np
import scipy.io as io
import tqdm, einops
from .common import *


def get_num_input_channels(tensor_weights):
    return tensor_weights.size()[1]


def get_ele_per_input_channel(tensor_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[:, 0, :, :].numel()


def get_input_channels(tensor_weights, st_id, ed_id):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[:, st_id:ed_id, :, :]


def assign_input_channels(tensor_weights, st_id, ed_id, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[:, st_id:ed_id, :, :] = quant_weights
    return


def get_num_output_channels(tensor_weights):
    return tensor_weights.size()[0]


def get_ele_per_output_channel(tensor_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    return tensor_weights[0, :, :, :].numel()


def get_output_channels(tensor_weights, st_id, ed_id):
    weights_copy = tensor_weights.clone()
    if len(weights_copy.shape) == 2:
        weights_copy = weights_copy[..., None, None]
    return weights_copy[st_id:ed_id, :, :, :]


def get_output_channels_inds(tensor_weights, inds):
    weights_copy = tensor_weights.clone()
    return weights_copy[inds]


def assign_output_channels(tensor_weights, st_id, ed_id, quant_weights):
    if len(tensor_weights.shape) == 2:
        tensor_weights = tensor_weights[..., None, None]
    tensor_weights[st_id:ed_id, :, :, :] = quant_weights


def assign_output_channels_inds(tensor_weights, inds, quant_weights):
    tensor_weights[inds, ...] = quant_weights


#####################


def pruning(data, amount, mode="unstructured", rank="l1"):
    if amount <= 0:
        return data
    elif amount >= 1:
        return torch.zeros_like(data, dtype=data.dtype, device=data.device)

    return data * get_mask(data, amount, rank=rank)


@torch.no_grad()
def get_mask(data, amount, rank="l1"):
    if rank == "l1":
        data = data.abs()

    mask = torch.ones_like(data).to(data.device)
    if amount <= 0:
        return mask
    assert amount <= 1
    k = int(amount * data.numel())
    if not k:
        print(k)
        return mask

    topk = torch.topk(data.view(-1), k=k, largest=False, sorted=False)
    mask.view(-1)[topk.indices] = 0
    return mask


def load_rd_curve(archname, layers, maxprunerates, datapath, y_tag="rd_dist"):
    nlayers = len(layers)
    rd_dist = []
    rd_phi = []

    for l in range(0, nlayers):
        rd_dist_l = []
        rd_phi_l = []

        matpath = "%s/%s_%03d.mat" % (datapath, archname, l)
        mat = io.loadmat(matpath)
        rd_dist_l.append(mat[y_tag][0])
        rd_phi_l.append(mat["rd_amount"][0])
        rd_dist.append(np.array(rd_dist_l))
        rd_phi.append(np.array(rd_phi_l))

    return rd_dist, rd_phi


def cal_total_num_weights(layers):
    nweights = 0
    nlayers = len(layers)

    for l in range(0, nlayers):
        n_filter_elements = layers[l].weight.numel()
        nweights += n_filter_elements
        # print('layer %d weights %d, %s' % (l, n_filter_elements, layers[l].weight.shape))
    return nweights


def closest_value_index(float_list, target):
    return np.argmin(np.abs(np.array(float_list) - target))



from multiprocessing import Pool, Array
import ctypes
from functools import partial
from contextlib import closing


def dp_n(surv, sorted_nweights_n, sorted_rd_phi_n, sorted_rd_dist_n, sorted_prev_pc_n, dp_n_1, n):
    dp_ = np.frombuffer(shared_arr.get_obj(), dtype=np.float32)
    g = np.frombuffer(shared_arr_2.get_obj(), dtype=np.uint32)
    allowed_phi_inds = ((surv > sorted_nweights_n * (1 - sorted_rd_phi_n)) & (sorted_rd_phi_n >= sorted_prev_pc_n)).nonzero()[0]
    if n == 0:
        g[surv] = closest_value_index(1 - sorted_rd_phi_n, surv / sorted_nweights_n)
        dp_[surv] = sorted_rd_dist_n[g[surv]]
        return
    f_all_n = sorted_rd_dist_n[allowed_phi_inds] + dp_n_1[surv - (sorted_nweights_n * (1 - sorted_rd_phi_n)[allowed_phi_inds]).astype(int)]
    g[surv] = int(np.argmin(f_all_n) + allowed_phi_inds[0])
    dp_[surv] = min(f_all_n)
    return


# def dp_pruning(layers, rd_dist, rd_phi, target_sparsity, prev_pc, g=None):
#     nlayers = len(layers)
#     nweights = np.array([layer.weight.numel() for layer in layers])
#     total_weights = sum(nweights)
#     sorted_layer_indices = np.argsort(nweights)
#     sorted_nweights = sorted(nweights)
#     sorted_rd_dist = [rd_dist[sort_idx][0] for sort_idx in sorted_layer_indices]
#     sorted_rd_phi = [rd_phi[sort_idx][0] for sort_idx in sorted_layer_indices]
#     sorted_prev_pc = [prev_pc[sort_idx][0] for sort_idx in sorted_layer_indices]
#     target_nweights = int((1-target_sparsity) * total_weights)
#     dp = np.ones((nlayers, target_nweights)) * float('inf')

#     # if g is None:
#     g = np.zeros((nlayers, target_nweights), dtype=np.uint32)  # result indices of rd_phi (pruning amounts)
        
#     def init(shared_arr_, shared_arr_2_):
#         global shared_arr, shared_arr_2
#         shared_arr = shared_arr_
#         shared_arr_2 = shared_arr_2_


#     for n in tqdm.tqdm(range(nlayers)):
#         dp[n, 0] = sorted_rd_dist[n][sorted_rd_phi[0] == 1] + (dp[n-1, 0] if n > 1 else 0)
#         g[n, 0] = closest_value_index(sorted_rd_phi[n], 1)

#         max_weights_n = int(min(sum(sorted_nweights[:n+1]), target_nweights))
#         shared_arr = Array(ctypes.c_float, max_weights_n)
#         shared_arr_2 = Array(ctypes.c_uint32, max_weights_n)

#         with closing(Pool(initializer=init, initargs=(shared_arr, shared_arr_2))) as p:
#             p.map_async(partial(dp_n, sorted_nweights_n=sorted_nweights[n],
#                                 sorted_rd_dist_n=sorted_rd_dist[n], 
#                                 sorted_rd_phi_n=sorted_rd_phi[n],
#                                 sorted_prev_pc_n=sorted_prev_pc[n],
#                                 n=n, dp_n_1=dp[n-1, :max_weights_n]), list(range(1, max_weights_n)))
#         p.join()
#         dp[n, 1: max_weights_n] = np.frombuffer(shared_arr.get_obj(), dtype=np.float32)[1:]
#         g[n, 1: max_weights_n] = np.frombuffer(shared_arr_2.get_obj(), dtype=np.uint32)[1:]
#         if max_weights_n < target_nweights:
#             dp[n, max_weights_n:] = dp[n, max_weights_n - 1]
#             g[n, max_weights_n:] = g[n, max_weights_n - 1]

#     dp_phi = [0 for _ in range(nlayers)]
#     survived_nweights = target_nweights
#     for sort_idx, n in zip(sorted_layer_indices[::-1], range(nlayers-1, -1, -1)):
#         dp_phi[sort_idx] = sorted_rd_phi[n][g[n, max(0, survived_nweights - 1)]]
#         survived_nweights =  max(0, survived_nweights - int((1 - dp_phi[sort_idx]) * sorted_nweights[n]))
#     return [[phi] for phi in dp_phi], dp, g

def closest_value_index_torch(float_list: torch.Tensor, target):
    return torch.argmin((float_list - target).abs())

def closest_value_index_2d(float_list: torch.Tensor, target):
    return torch.argmin((float_list - target).abs(), dim=1)

def dp_pruning(layers, rd_dist, rd_phi, target_sparsity, prev_pc, g=None, device="cpu", piece_length=4096):
    device = torch.device(device)

    nlayers = len(layers)
    nweights = np.array([layer.weight.numel() for layer in layers])
    total_weights = sum(nweights)
    sorted_layer_indices = np.argsort(nweights)
    sorted_nweights = sorted(nweights)
    sorted_rd_dist = [torch.tensor(rd_dist).to(device)[sort_idx][0] for sort_idx in sorted_layer_indices]
    sorted_rd_phi = [torch.tensor(rd_phi).to(device)[sort_idx][0] for sort_idx in sorted_layer_indices]
    sorted_prev_pc = [torch.tensor(prev_pc).to(device)[sort_idx][0] for sort_idx in sorted_layer_indices]
    target_nweights = int((1-target_sparsity) * total_weights)

    dp = torch.ones((nlayers, target_nweights), device=device) * float('inf')

    if g is None:
        g = torch.zeros((nlayers, target_nweights), dtype=torch.long, device=device)  # result indices of rd_phi (pruning amounts)
        inf_tensor = torch.tensor([float("inf")], device=device)

        survs = torch.tensor(list(range(target_nweights)), dtype=torch.long, device=device)
        survs = einops.repeat(survs, "n -> n p", p=max(len(sorted_rd_phi[n]) for n in range(nlayers)))

        for n in tqdm.tqdm(range(nlayers)):
            dp[n, 0] = sorted_rd_dist[n][sorted_rd_phi[0] == 1] + (dp[n-1, 0] if n > 1 else 0)
            g[n, 0] = closest_value_index_torch(sorted_rd_phi[n], 1)

            max_weights_n = min(sum(sorted_nweights[:n+1]), target_nweights)

            
            
            num_pieces = torch.tensor((max_weights_n - 1) / piece_length).ceil().int().item()
            for ip in range(num_pieces):
                start = 1 + ip * piece_length
                end = min(target_nweights, (ip + 1) * piece_length)
                survs_ = survs[start: end, :len(sorted_rd_phi[n])]
                sorted_rd_phi_ = einops.repeat(sorted_rd_phi[n], "p -> n p", n=len(survs_))
                sorted_rd_dist_ = einops.repeat(sorted_rd_dist[n], "p -> n p", n=len(survs_))

                allowed_phi_masks = ((survs_ > sorted_nweights[n] * (1 - sorted_rd_phi_)) & (sorted_rd_phi_ >= sorted_prev_pc[n]))
                if n == 0:
                    g[n, start: end] = closest_value_index_2d(1 - sorted_rd_phi_, (survs_[:, 0] / sorted_nweights[n])[:, None])
                    dp[n, start: end] = sorted_rd_dist[n][g[n, start: end]]
                else:
                    f_all_n = torch.where(allowed_phi_masks, sorted_rd_dist_ + dp[n-1, survs_ - (sorted_nweights[n] * (1 - sorted_rd_phi_)).long()], inf_tensor) 
                    g[n, start: end] = torch.argmin(f_all_n, dim=1)
                    dp[n, start: end] = torch.min(f_all_n, dim=1)[0]
                    del f_all_n

            if max_weights_n < target_nweights:
                dp[n, max_weights_n:] = dp[n, max_weights_n - 1]
                g[n, max_weights_n:] = g[n, max_weights_n - 1]
            
            del allowed_phi_masks, survs_

    dp_phi = [0 for _ in range(nlayers)]
    survived_nweights = target_nweights
    for sort_idx, n in zip(sorted_layer_indices[::-1], range(nlayers-1, -1, -1)):
        dp_phi[sort_idx] = sorted_rd_phi[n][g[n, max(0, survived_nweights - 1)]]
        survived_nweights =  max(0, survived_nweights - int((1 - dp_phi[sort_idx]) * sorted_nweights[n]))
    return [[phi] for phi in dp_phi], dp, g
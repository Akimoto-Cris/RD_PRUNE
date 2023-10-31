import torch
from torch.nn.utils import prune
from tools.utils import get_weights, get_modules
import numpy as np
import tools.common as common
import tools.algo as algo
import os
import scipy.io as io
import pickle

def weight_pruner_loader(pruner_string):
    """
    Gives you the pruning methods: LAMP, Glob, Unif, Unif+, ERK and RD
    """
    if pruner_string == "lamp":
        return prune_weights_lamp
    elif pruner_string == "glob":
        return prune_weights_global
    elif pruner_string == "unif":
        return prune_weights_uniform
    elif pruner_string == "unifplus":
        return prune_weights_unifplus
    elif pruner_string == "erk":
        return prune_weights_erk
    elif pruner_string == "rd":
        return RDPruner()
    else:
        raise ValueError("Unknown pruner")


"""
prune_weights_reparam: Allocate identity mask to every weight tensors.
prune_weights_l1predefined: Perform layerwise pruning w.r.t. given amounts.
"""



def prune_weights_reparam(model):
    module_list = get_modules(model)
    for m in module_list:
        prune.identity(m, name="weight")


def prune_weights_l1predefined(model, amounts, only_layerids=None):
    mlist = get_modules(model)
    for idx, m in enumerate(mlist):
        if only_layerids is not None and idx not in only_layerids:
            continue
        prune.l1_unstructured(m, name="weight", amount=float(amounts[idx]))


def prune_weights_l1structured(model, amounts, only_layerids=None):
    mlist = get_modules(model)
    for idx, m in enumerate(mlist):
        if only_layerids is not None and idx not in only_layerids:
            continue
        prune.ln_structured(m, name="weight", amount=float(amounts[idx]), n=1, dim=1)


"""
Methods: All weights
"""

def gen_rd_curves(net, loader, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = "./%s_ndz_%04d_rdcurves_channelwise_opt_dist" % (args.model, args.maxsps)
    else:
        path_output = "%s/%s_ndz_%04d_rdcurves_channelwise_opt_dist/%s/" % (
            prefix,
            args.model,
            args.maxsps,
            suffix,
        )

    layers = common.findconv(net, False)
    hookedlayers = common.hooklayers(layers)
    if args.dataset == "cifar":
        dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    else:
        dummy_input = torch.zeros((1, 3, 224, 224)).cuda()

    _ = net(dummy_input)
    fil = [hasattr(h, "output") for h in hookedlayers]
    if False in fil:
        layers = [layers[i] for i in range(len(layers)) if fil[i]]
    for l in hookedlayers:
        l.close()

    print("total number of layers: %d" % (len(layers)))
    print(f"saving curves to {path_output}")
    isExists = os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        return algo.load_rd_curve(args.model, layers, args.maxsps, path_output)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
    if args.dataset == "cifar":
        Y, labels = common.predict2_withgt(net, loader)
    else:
        Y, labels = common.predict_dali_withgt(net, loader)

    top_1, top_5 = common.accuracy(Y, labels, topk=(1, 5))
    print("original network %s accuracy on calibration set: top1 %5.2f top5 %5.2f" % (args.model, top_1, top_5))

    with torch.no_grad():
        for layerid in range(len(layers)):
            print(f"generating curves for layer-{layerid}")
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxsps + 1, device=torch.device("cuda:0"))
            rst_dist = torch.ones(args.maxsps + 1, device=torch.device("cuda:0"))
            min_amount = 0

            for d in range(args.maxsps + 1):
                amount = (1.0 - min_amount) * d / args.maxsps + min_amount      # amount to get pruned
                rst_amount[d] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount)

                layers[layerid].weight.data = prune_weights

                if args.dataset == "cifar":
                    Y_hat = common.predict2(net, loader)
                else:
                    Y_hat = common.predict_dali(net, loader)

                if args.worst_case_curve:
                    cur_dist = ((Y - Y_hat) ** 2).mean(dim=1).max() 
                else:
                    cur_dist = ((Y - Y_hat) ** 2).mean()

                rst_dist[d] = cur_dist
                layers[layerid].weight.data = layer_weights

            io.savemat(
                ("%s/%s_%03d.mat" % (path_output, args.model, layerid)),
                {
                    "rd_amount": rst_amount.cpu().numpy(),
                    "rd_dist": rst_dist.cpu().numpy(),
                }
            )

    return algo.load_rd_curve(args.model, layers, args.maxsps, path_output)


def gen_rd_curves_synth_data(net, args, prefix=None, suffix=None):
    if prefix is None:
        path_output = "./%s_ndz_%04d_rdcurves_channelwise_opt_dist_synth" % (args.model, args.maxsps)
    else:
        path_output = "%s/%s_ndz_%04d_rdcurves_channelwise_opt_dist_synth/%s/" % (
            prefix,
            args.model,
            args.maxsps,
            suffix,
        )

    layers = common.findconv(net, False)
    print("total number of layers: %d" % (len(layers)))
    print(f"saving curves to {path_output}")
    isExists = os.path.exists(path_output)
    if not isExists:
        os.makedirs(path_output)
    elif len(os.listdir(path_output)) == len(layers):
        print("found curves in", path_output)
        return algo.load_rd_curve(args.model, layers, args.maxsps, path_output)

    for l in range(0, len(layers)):
        layer_weights = layers[l].weight.clone()
        nchannels = algo.get_num_output_channels(layer_weights)
        n_channel_elements = algo.get_ele_per_output_channel(layer_weights)

    net.eval()
    if args.dataset == "cifar":
        X = torch.normal(torch.zeros(args.calib_size, 3, 32, 32).cuda(), torch.ones(args.calib_size, 3, 32, 32).cuda())
    else:
        X = torch.normal(torch.zeros(args.calib_size, 3, 224, 224).cuda(), torch.ones(args.calib_size, 3, 224, 224).cuda())
    
    Y = common.predict_tensor(net, X, 256)

    with torch.no_grad():
        for layerid in range(len(layers)):
            print(f"generating curves for layer-{layerid}")
            layer_weights = layers[layerid].weight.clone()

            rst_amount = torch.ones(args.maxsps + 1).cuda()
            rst_dist = torch.ones(args.maxsps + 1).cuda()

            min_amount = 0

            for d in range(args.maxsps + 1):
                amount = (1.0 - min_amount) * d / args.maxsps + min_amount
                rst_amount[d] = amount
                prune_weights = algo.pruning(layers[layerid].weight, amount)

                layers[layerid].weight.data = prune_weights

                Y_hat = common.predict_tensor(net, X, 256)
                if args.worst_case_curve:
                    cur_dist = ((Y - Y_hat) ** 2).mean(dim=1).max()
                else:
                    cur_dist = ((Y - Y_hat) ** 2).mean()

                rst_dist[d] = cur_dist
                layers[layerid].weight.data = layer_weights

            io.savemat(
                ("%s/%s_%03d.mat" % (path_output, args.model, layerid)),
                {
                    "rd_amount": rst_amount.cpu().numpy(),
                    "rd_dist": rst_dist.cpu().numpy(),
                },
            )

    return algo.load_rd_curve(args.model, layers, args.maxsps, path_output)


class RDPruner:
    def __call__(self, model, amount, args, loader, container, to_prune_layerids=None, epoch_cnt=0):
        if not hasattr(self, "amount"):  # initialize at first iter
            assert amount <= 1
            self.amount = amount
            self.iter_cnt = args.iter_start
            unmaskeds = _count_unmasked_weights(model)
            totals = _count_total_weights(model)
            self.prev_pc = [[1 - float(surv / tot)] for surv, tot in zip(unmaskeds, totals)]

        sd = model.state_dict()
        new = sd.copy()
        for k, v in sd.items():
            if "weight_orig" in k:
                new[k.replace("weight_orig", "weight")] = v * sd[k.replace("weight_orig", "weight_mask")]

        container.load_state_dict(new, strict=False)  #
        if not hasattr(self, "layers"):
            self.layers = common.findconv(container, False)
        target_sparsity = 1.0 - (1.0 - self.amount) ** self.iter_cnt

        print("Generating RD Curves...")
        if args.synth_data:
            rd_dist, rd_phi = gen_rd_curves_synth_data(
                container,
                args,
                prefix=f"./rd_retrain/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/",
                suffix=f"sp{target_sparsity}",
            )
        else:
            rd_dist, rd_phi = gen_rd_curves(
                container,
                loader,
                args,
                prefix=f"./rd_retrain/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/ranking_{args.ranking}/",
                suffix=f"sp{target_sparsity}",
            )

        print("SOLVING LAYER-WISE SPARSITY")
        dp_save_path = f"./rd_retrain/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/sp{target_sparsity}_{args.model}_ndz_{args.maxsps:04d}_rdcurves_ranking_{args.ranking}_opt_dist_dp.pkl"
        g = None
        if os.path.exists(dp_save_path):
            p = pickle.load(open(dp_save_path, "rb"))
            g = p["g"]
        
        pc_phi, dp, g = algo.dp_pruning(self.layers, rd_dist, rd_phi, target_sparsity, self.prev_pc, g=g)
        
        amounts = [torch.Tensor([max(0, 1 - (1 - p[0]) / (1 - pp[0]))])[0].cuda() for p, pp in zip(pc_phi, self.prev_pc)]
        print('\n'.join(f"layer-{l}: surv {pp[0] * 100:.1f}% -> {p[0] * 100:.1f}%" for l, (p, pp) in enumerate(zip(pc_phi, self.prev_pc))), "\n")
        self.prev_pc = pc_phi
        
        if not os.path.exists(dp_save_path):
            with open(dp_save_path,"wb") as f:
                pickle.dump({"dp": dp, "g": g}, f)
        self.amounts = amounts
        if args.prune_mode == "structured":
            prune_weights_l1structured(model, amounts, to_prune_layerids)
        else:
            prune_weights_l1predefined(model, amounts, to_prune_layerids)

        mask_save_path = f"./rd_retrain/weight_rewind_{args.weight_rewind}/{args.seed}/remask_per_iter_{args.remask_per_iter}/sp{target_sparsity}_{args.model}_ndz_{args.maxsps:04d}_rdcurves_ranking_{args.ranking}_opt_dist_mask.pt"
        to_save = {k: v for k, v in model.state_dict().items() if "weight_mask" in k}
        torch.save(to_save, mask_save_path)
        
        self.iter_cnt += 1



def prune_weights_global(model, amount):
    parameters_to_prune = _extract_weight_tuples(model)
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


def prune_weights_lamp(model, amount):
    assert amount <= 1
    amounts = _compute_lamp_amounts(model, amount)
    print(amounts)
    prune_weights_l1predefined(model, amounts)


def prune_weights_uniform(model, amount):
    module_list = get_modules(model)
    assert amount <= 1  # Can be updated later to handle > 1.
    for m in module_list:
        prune.l1_unstructured(m, name="weight", amount=amount)


def prune_weights_unifplus(model, amount):
    assert amount <= 1
    amounts = _compute_unifplus_amounts(model, amount)
    prune_weights_l1predefined(model, amounts)


def prune_weights_erk(model, amount):
    assert amount <= 1
    amounts = _compute_erk_amounts(model, amount)
    prune_weights_l1predefined(model, amounts)


def prune_weights_rd(model, amount, *args, **kwargs):
    assert amount <= 1
    amounts = _compute_rd_amounts(model, amount, *args, **kwargs)
    print(amounts)
    prune_weights_l1predefined(model, amounts)


"""
These are not intended to be exported.
"""


def _extract_weight_tuples(model):
    """
    Gives you well-packed weight tensors for global pruning.
    """
    mlist = get_modules(model)
    return tuple([(m, "weight") for m in mlist])


def _compute_unifplus_amounts(model, amount):
    """
    Compute # of weights to prune in each layer.
    """
    amounts = []
    wlist = get_weights(model)
    unmaskeds = _count_unmasked_weights(model)
    totals = _count_total_weights(model)

    last_layer_minimum = np.round(totals[-1] * 0.2)  # Minimum number of last-layer weights to keep
    total_to_prune = np.round(unmaskeds.sum() * amount)

    if wlist[0].dim() == 4:
        amounts.append(0)  # Leave the first layer unpruned.
        frac_to_prune = (total_to_prune * 1.0) / (unmaskeds[1:].sum())
        if frac_to_prune > 1.0:
            raise ValueError("Cannot be pruned further by the Unif+ scheme! (first layer exception)")
        last_layer_to_surv_planned = np.round((1.0 - frac_to_prune) * unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune - last_layer_to_prune) * 1.0) / (unmaskeds[1:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (first+last layer exception)")
            amounts.extend([frac_to_prune_middle] * (unmaskeds.size(0) - 2))
            amounts.append((last_layer_to_prune * 1.0) / unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune] * (unmaskeds.size(0) - 1))
    else:
        frac_to_prune = (total_to_prune * 1.0) / (unmaskeds.sum())
        last_layer_to_surv_planned = np.round((1.0 - frac_to_prune) * unmaskeds[-1])
        if last_layer_to_surv_planned < last_layer_minimum:
            last_layer_to_prune = unmaskeds[-1] - last_layer_minimum
            frac_to_prune_middle = ((total_to_prune - last_layer_to_prune) * 1.0) / (unmaskeds[:-1].sum())
            if frac_to_prune_middle > 1.0:
                raise ValueError("Cannot be pruned further by the Unif+ scheme! (last layer exception)")
            amounts.extend([frac_to_prune_middle] * (unmaskeds.size(0) - 1))
            amounts.append((last_layer_to_prune * 1.0) / unmaskeds[-1])
        else:
            amounts.extend([frac_to_prune] * (unmaskeds.size(0)))
    return amounts


def _compute_erk_amounts(model, amount):
    unmaskeds = _count_unmasked_weights(model)
    erks = _compute_erks(model)

    return _amounts_from_eps(unmaskeds, erks, amount)


def _amounts_from_eps(unmaskeds, ers, amount):
    num_layers = ers.size(0)
    layers_to_keep_dense = torch.zeros(num_layers)
    total_to_survive = (1.0 - amount) * unmaskeds.sum()  # Total to keep.

    # Determine some layers to keep dense.
    is_eps_invalid = True
    while is_eps_invalid:
        unmasked_among_prunables = (unmaskeds * (1 - layers_to_keep_dense)).sum()
        to_survive_among_prunables = total_to_survive - (layers_to_keep_dense * unmaskeds).sum()

        ers_of_prunables = ers * (1.0 - layers_to_keep_dense)
        survs_of_prunables = torch.round(to_survive_among_prunables * ers_of_prunables / ers_of_prunables.sum())

        layer_to_make_dense = -1
        max_ratio = 1.0
        for idx in range(num_layers):
            if layers_to_keep_dense[idx] == 0:
                if survs_of_prunables[idx] / unmaskeds[idx] > max_ratio:
                    layer_to_make_dense = idx
                    max_ratio = survs_of_prunables[idx] / unmaskeds[idx]

        if layer_to_make_dense == -1:
            is_eps_invalid = False
        else:
            layers_to_keep_dense[layer_to_make_dense] = 1

    amounts = torch.zeros(num_layers)

    for idx in range(num_layers):
        if layers_to_keep_dense[idx] == 1:
            amounts[idx] = 0.0
        else:
            amounts[idx] = 1.0 - (survs_of_prunables[idx] / unmaskeds[idx])
    return amounts


def _compute_lamp_amounts(model, amount):
    """
    Compute normalization schemes.
    """
    unmaskeds = _count_unmasked_weights(model)
    num_surv = int(np.round(unmaskeds.sum() * (1.0 - amount)))

    flattened_scores = [_normalize_scores(w**2).view(-1) for w in get_weights(model)]
    concat_scores = torch.cat(flattened_scores, dim=0)
    topks, _ = torch.topk(concat_scores, num_surv)
    threshold = topks[-1]

    # We don't care much about tiebreakers, for now.
    final_survs = [
        torch.ge(score, threshold * torch.ones(score.size()).to(score.device)).sum() for score in flattened_scores
    ]
    amounts = []
    for idx, final_surv in enumerate(final_survs):
        amounts.append(1.0 - (final_surv / unmaskeds[idx]))

    return amounts


def _compute_erks(model):
    wlist = get_weights(model)
    erks = torch.zeros(len(wlist))
    for idx, w in enumerate(wlist):
        if w.dim() == 4:
            erks[idx] = w.size(0) + w.size(1) + w.size(2) + w.size(3)
        else:
            erks[idx] = w.size(0) + w.size(1)
    return erks


@torch.no_grad()
def _count_unmasked_weights(model):
    """
    Return a 1-dimensional tensor of #unmasked weights.
    """
    mlist = get_modules(model)
    unmaskeds = []
    for m in mlist:
        unmaskeds.append(m.weight.count_nonzero())
    return torch.FloatTensor(unmaskeds)


@torch.no_grad()
def _count_total_weights(model):
    """
    Return a 1-dimensional tensor of #total weights.
    """
    wlist = get_weights(model)
    numels = []
    for w in wlist:
        numels.append(w.numel())
    return torch.FloatTensor(numels)


def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores, sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape, device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[: len(scores_cumsum_temp) - 1]
    # normalize by cumulative sum
    sorted_scores /= scores.sum() - scores_cumsum
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape, device=scores.device)
    new_scores[sorted_idx] = sorted_scores

    return new_scores.view(scores.shape)

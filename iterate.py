import torch, argparse, random, os
import numpy as np
from tools import *
import tools.common as common

""" ARGUMENT PARSING """
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--cuda", type=int, help="cuda number")
parser.add_argument("--model", type=str, help="network")
parser.add_argument("--pruner", type=str, help="pruning method")
parser.add_argument("--dataset", type=str, choices=["cifar", "imagenet"], default="cifar")
parser.add_argument("--iter_start", type=int, default=1, help="start iteration for pruning (set >1 for resume)")
parser.add_argument("--iter_end", type=int, default=20, help="end iteration for pruning")
parser.add_argument("--maxsps", type=int, default=100)
parser.add_argument("--ranking", type=str, default="l1")
parser.add_argument(
    "--prune_mode", "-pm", type=str, default="unstructured", choices=["unstructured", "structured"]
)
parser.add_argument("--calib_size", type=int, default=20)
parser.add_argument("--weight_rewind", action="store_true")
parser.add_argument("--worst_case_curve", "-wcc", action="store_true")
parser.add_argument("--synth_data", action="store_true")
parser.add_argument("--singlelayer", action="store_true")
parser.add_argument(
    "--flop_budget",
    action="store_true",
    help="use flop as the targeting budget in ternary search instead of sparsity. if true, `amounts` and `target_sparsity` variables in the codes will represent flops instead",
)
args = parser.parse_args()

""" SET THE SEED """
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

DEVICE = args.cuda

""" IMPORT LOADERS/MODELS/PRUNERS/TRAINERS"""
model, amount_per_it, batch_size, opt_pre, opt_post = model_and_opt_loader(
    args.model, DEVICE, weight_rewind=args.weight_rewind, reparam=True
)
train_loader, test_loader, calib_loader = dataset_loader(args.model, batch_size=batch_size, args=args)
pruner = weight_pruner_loader(args.pruner)
trainer = trainer_loader(args)
container, _, _, _, _ = model_and_opt_loader(args.model, DEVICE, reparam=False, weight_rewind=args.weight_rewind)
""" SET SAVE PATHS """
DICT_PATH = f"./dicts/{args.model}/{args.seed}/{args.prune_mode}/"
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)
BASE_PATH = f"./results/iterate/{args.model}/{args.seed}/{args.prune_mode}/"
if args.weight_rewind:
    BASE_PATH += "/weight_rewind/"
    DICT_PATH += "/weight_rewind/"
if not os.path.exists(BASE_PATH):
    os.makedirs(BASE_PATH)
if not os.path.exists(DICT_PATH):
    os.makedirs(DICT_PATH)

results_to_save = []

""" PRETRAIN (IF NEEDED) """
if args.iter_start == 1:
    filename_string = "unpruned.pth.tar"
else:
    filename_string = args.pruner + str(args.iter_start - 1) + ".pth.tar"
if os.path.exists(os.path.join(DICT_PATH, filename_string)):
    print(f"LOADING PRE-TRAINED MODEL: SEED: {args.seed}, MODEL: {args.model}, ITER: {args.iter_start - 1}")
    state_dict = torch.load(os.path.join(DICT_PATH, filename_string), map_location=torch.device(DEVICE))
    model.load_state_dict(state_dict)
    try:
        results_to_save = [[i.item() for i in l] for l in torch.load(BASE_PATH + f"/{args.pruner}.tsr")]
    except:
        pass
else:
    if args.iter_start == 1:
        print(f"PRE-TRAINING A MODEL: SEED: {args.seed}, MODEL: {args.model}")
        pretrain_results = trainer(model, opt_pre, train_loader, test_loader, print_steps=1000)
        torch.save(pretrain_results, DICT_PATH + "/unpruned_loss.dtx")
        torch.save(model.state_dict(), os.path.join(DICT_PATH, "unpruned.pth.tar"))
    else:
        raise ValueError("No (iteratively pruned/trained) model found!")

epoch_cnt = args.iter_start * opt_post["steps"]
# opt_post["cutmix_alpha"] = args.cutmix_alpha
# print(args.cutmix_alpha)

""" PRUNE AND RETRAIN """
for it in range(args.iter_start, args.iter_end + 1):
    print(f"Pruning for iteration {it}: METHOD: {args.pruner}")
    flops = common.get_model_flops(model, args.dataset)
    print(f"Before prune: FLOPs: {flops}")
    if args.pruner == "rd":
        args.remask_per_iter = opt_post["steps"]  # just for naming
        pruner(model, amount_per_it, args, calib_loader, container, epoch_cnt=epoch_cnt)
    else:
        pruner(model, amount_per_it)

    if args.weight_rewind and os.path.exists(os.path.join(DICT_PATH, args.pruner + str(it - 1) + ".pth.tar")):
        model.load_state_dict(
            {
                k: v
                for k, v in torch.load(os.path.join(DICT_PATH, args.pruner + str(it - 1) + ".pth.tar")).items()
                if "mask" not in k
            },
            strict=False,
        )

    flops = common.get_model_flops(model, args.dataset)
    sparse = utils.get_model_sparsity(model)
    print(f"sparsity: {sparse * 100} (%)")
    print(f"Remained FLOPs: {flops * 100} (%)")

    result_log = trainer(model, opt_post, train_loader, test_loader, print_steps=100)
    result_log.append(get_model_sparsity(model))
    result_log.append(flops)
    results_to_save.append(result_log)
    torch.save(torch.FloatTensor(results_to_save), BASE_PATH + f"/{args.pruner}.tsr")
    # if args.weight_rewind:
    torch.save(model.state_dict(), os.path.join(DICT_PATH, args.pruner + str(it) + ".pth.tar"))

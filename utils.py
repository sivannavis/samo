import os
import eval_metrics as em
import torch
import random
import numpy as np
from itertools import count
from math import cos, gamma, pi, sin, sqrt
from typing import Callable, Iterator, List
import matplotlib.pyplot as plt

# Seeding
def seed_worker(worker_id):
    """
    Used in generating seed for the worker of torch.utils.data.Dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_seed(random_seed, cudnn_deterministic=True):
    """ set_random_seed(random_seed, cudnn_deterministic=True)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      cudnn_deterministic: for torch.backends.cudnn.deterministic

    Note: this default configuration may result in RuntimeError
    see https://pytorch.org/docs/stable/notes/randomness.html
    """

    # # initialization
    torch.manual_seed(random_seed) #PyTorch random number generator
    random.seed(random_seed) #python operators
    np.random.seed(random_seed) # numpy generator
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = cudnn_deterministic # CUDA convolution determinism
        torch.backends.cudnn.benchmark = False # Disabling the benchmarking feature, deterministically select an algorithm

# Learning rate schedules
def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def adjust_learning_rate(args, lr, optimizer, epoch_num):
    '''
    Exponential learning rate schedule
    '''
    lr = lr * (args.lr_decay ** (epoch_num // args.interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Compute eval metrics
def compute_eer_tdcf(args, cm_score_file):
    asv_score_file = os.path.join('scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt')

    # Fix tandem detection cost function (t-DCF) parameters
    Pspoof = 0.05
    cost_model = {
        'Pspoof': Pspoof,  # Prior probability of a spoofing attack
        'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
        'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
        'Cmiss_asv': 1,  # Cost of ASV system falsely rejecting target speaker
        'Cfa_asv': 10,  # Cost of ASV system falsely accepting nontarget speaker
        'Cmiss_cm': 1,  # Cost of CM system falsely rejecting target speaker
        'Cfa_cm': 10,  # Cost of CM system falsely accepting spoof
    }

    # Load organizers' ASV scores
    asv_data = np.genfromtxt(asv_score_file, dtype=str)
    # asv_sources = asv_data[:, 0]
    asv_keys = asv_data[:, 1]
    asv_scores = asv_data[:, 2].astype(float)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    # cm_utt_id = cm_data[:, 0]
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2].astype(int) # label
    cm_scores = cm_data[:, 3].astype(float)

    # Extract target, nontarget, and spoof scores from the ASV scores
    tar_asv = asv_scores[asv_keys == 'target']
    non_asv = asv_scores[asv_keys == 'nontarget']
    spoof_asv = asv_scores[asv_keys == 'spoof']

    # Extract bona fide (real human) and spoof scores from the CM scores
    bona_cm = cm_scores[cm_keys == 0]
    spoof_cm = cm_scores[cm_keys == 1]

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    [Pfa_asv, Pmiss_asv, Pmiss_spoof_asv] = em.obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold)

    # Compute t-DCF
    tDCF_curve, CM_thresholds = em.compute_tDCF(bona_cm, spoof_cm, Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, cost_model)

    # Minimum t-DCF
    min_tDCF_index = np.argmin(tDCF_curve)
    min_tDCF = tDCF_curve[min_tDCF_index]

    # test individual attacks
    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    eer_cm_lst = {}
    for attack in attack_types:
        if attack == "-":
            continue
        # Extract bona fide (real human) and spoof scores from the CM scores
        bona_cm = cm_scores[cm_keys == 0]
        spoof_cm = cm_scores[cm_sources == attack]

        # EERs of the standalone systems and fix ASV operating point to EER threshold
        eer_att = em.compute_eer(bona_cm, spoof_cm)[0]
        if not np.isnan(eer_att):
            eer_cm_lst[attack] = eer_att
            # eer_cm_lst.append("{:5.2f} %".format(eer_cm * 100))
        else:
            continue

    output_file = "./breakdown/{}.txt".format(args.save_score)
    with open(output_file, "w") as f_res:
        f_res.write('\nCM SYSTEM\n')
        f_res.write('\tEER\t\t= {:8.9f} % '
                    '(Equal error rate for countermeasure)\n'.format(
            eer_cm * 100))

        f_res.write('\nTANDEM\n')
        f_res.write('\tmin-tDCF\t\t= {:8.9f}\n'.format(min_tDCF))

        f_res.write('\nBREAKDOWN CM SYSTEM\n')
        for attack_type in attack_types:
            _eer = eer_cm_lst[attack_type] * 100
            f_res.write(
                f'\tEER {attack_type}\t\t= {_eer:8.9f} % \n'
            )
    os.system(f"cat {output_file}")

    return eer_cm, min_tDCF

## Generate evenly distributed samples on a hypersphere
## Refer to https://stackoverflow.com/questions/57123194/how-to-distribute-points-evenly-on-the-surface-of-hyperspheres-in-higher-dimensi
def int_sin_m(x: float, m: int) -> float:
    """Computes the integral of sin^m(t) dt from 0 to x recursively"""
    if m == 0:
        return x
    elif m == 1:
        return 1 - cos(x)
    else:
        return (m - 1) / m * int_sin_m(x, m - 2) - cos(x) * sin(x) ** (
            m - 1
        ) / m

def primes() -> Iterator[int]:
    """Returns an infinite generator of prime numbers"""
    yield from (2, 3, 5, 7)
    composites = {}
    ps = primes()
    next(ps)
    p = next(ps)
    assert p == 3
    psq = p * p
    for i in count(9, 2):
        if i in composites:  # composite
            step = composites.pop(i)
        elif i < psq:  # prime
            yield i
            continue
        else:  # composite, = p*p
            assert i == psq
            step = 2 * p
            p = next(ps)
            psq = p * p
        i += step
        while i in composites:
            i += step
        composites[i] = step

def inverse_increasing(
    func: Callable[[float], float],
    target: float,
    lower: float,
    upper: float,
    atol: float = 1e-10,
) -> float:
    """Returns func inverse of target between lower and upper

    inverse is accurate to an absolute tolerance of atol, and
    must be monotonically increasing over the interval lower
    to upper
    """
    mid = (lower + upper) / 2
    approx = func(mid)
    while abs(approx - target) > atol:
        if approx > target:
            upper = mid
        else:
            lower = mid
        mid = (upper + lower) / 2
        approx = func(mid)
    return mid

def uniform_hypersphere(d: int, n: int) -> List[List[float]]:
    """Generate n points over the d dimensional hypersphere"""
    assert d > 1
    assert n > 0
    points = [[1 for _ in range(d)] for _ in range(n)]
    for i in range(n):
        t = 2 * pi * i / n
        points[i][0] *= sin(t)
        points[i][1] *= cos(t)
    for dim, prime in zip(range(2, d), primes()):
        offset = sqrt(prime)
        mult = gamma(dim / 2 + 0.5) / gamma(dim / 2) / sqrt(pi)

        def dim_func(y):
            return mult * int_sin_m(y, dim - 1)

        for i in range(n):
            deg = inverse_increasing(dim_func, i * offset % 1, 0, pi)
            for j in range(dim):
                points[i][j] *= sin(deg)
            points[i][dim] *= cos(deg)
    return points

# Generate experiments curves
def compare_exps(exp_dirs, root_dir, max_train_loss=None, max_dev_loss=None, eval_available=False):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))
    for folder in exp_dirs:
        out_fold = os.path.join(root_dir, folder)
        train_log_file = os.path.join(out_fold, "train_loss.log")
        dev_log_file = os.path.join(out_fold, "dev_loss.log")
        with open(train_log_file, "r") as train_log:
            x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in train_log.readlines()[1:]])
            it_per_batch = int(x[:, 1].max()) + 1
            x = x[it_per_batch - 1::it_per_batch]
            ax1.plot(range(1, len(x) + 1), x[:, 2])
            #     ax1.set_xticks(np.where(x[:,1]==0)[0][::10])
            #     ax1.set_xticklabels(np.arange(np.sum(x[:,1]==0))*10)
            if not max_train_loss is None:
                ax1.set_ylim([0, max_train_loss])
            ax1.set_title("Training Loss")
            ax1.grid()
            ax1.legend(exp_dirs)
        with open(dev_log_file, "r") as dev_log:
            x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in dev_log.readlines()[1:]])
            ax2.plot(range(1, len(x) + 1), x[:, 1])
            ax2.set_title("Validation Loss")
            if not max_dev_loss is None:
                ax2.set_ylim([0, max_dev_loss])
            ax2.legend(exp_dirs)

            ax3.plot(range(1, len(x) + 1), x[:, 2])
            ax3.set_title("Validation EER")
            ax3.minorticks_on()
            ax3.grid(b=True, which='major', linestyle='-')
            ax3.grid(b=True, which='minor', linestyle=':')
            ax3.set_ylim([0, 0.01])
            ax3.legend(exp_dirs)
        if eval_available:
            with open(os.path.join(out_fold, "test_loss.log"), "r") as eval_eer:
                x = np.array([[float(i) for i in line[0:-1].split('\t')] for line in eval_eer.readlines()[1:]])
                ax4.plot(range(1, len(x) + 1), x[:, 2])
                ax4.set_title("Test EER")
                ax4.minorticks_on()
                ax4.grid(b=True, which='major', linestyle='-')
                ax4.grid(b=True, which='minor', linestyle=':')
                ax4.set_ylim([0, 0.08])
                ax4.legend(exp_dirs)
    plt.show(block = True)
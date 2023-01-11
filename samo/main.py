import json
import shutil
import logging
import argparse
import os.path
from importlib import import_module
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from torch import nn, Tensor
from torch.utils.data import DataLoader, Subset
from torchcontrib.optim import SWA

from loss import SAMO, OCSoftmax
from utils import setup_seed, seed_worker, cosine_annealing, adjust_learning_rate, em, compute_eer_tdcf
from aasist.data_utils import genSpoof_list, ASVspoof2019_speaker_raw


def initParams():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--seed', type=int, help="random number seed", default=10)

    # Data folder prepare
    parser.add_argument("-d", "--path_to_database", type=str, help="dataset path",
                        default='/data2/sivan/')
    parser.add_argument("-p", "--path_to_protocol", type=str, help="protocol path",
                        default='./protocols/')
    parser.add_argument("-o", "--out_fold", type=str, help="output folder", required=False, default='./models/try/')
    parser.add_argument("--overwrite", action='store_true', help="overwrite output folder")

    # Dataset prepare
    parser.add_argument("--enc_dim", type=int, help="encoding dimension", default=160)

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=23, help="Mini batch size for training")
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--lr_min', type=float, default=0.000005, help="min learning rate in cosine annealing")
    parser.add_argument('--lr_decay', type=float, default=0.95, help="decay learning rate for exponential schedule")
    parser.add_argument('--interval', type=int, default=1, help="interval to decay lr for exponential schedule")
    parser.add_argument("--scheduler", type=str, default="cosine2", choices=["cosine", "cosine2", "exp", "clr"])
    parser.add_argument('--beta_1', type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument('--eps', type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers")

    # Loss setups
    parser.add_argument('-l', '--loss', type=str, default="ocsoftmax",
                        choices=["softmax", "ocsoftmax", "samo"], help="loss for training")
    parser.add_argument('--num_centers', type=int, default=20,
                        help="number of centers for the sub-center one-class loss")
    parser.add_argument('--initialize_centers', type=str, default="one_hot",
                        choices=["randomly", "evenly", "one_hot", "uniform"])
    parser.add_argument('--m_real', type=float, default=0.7, help="m_real for ocsoftmax/samo loss")
    parser.add_argument('--m_fake', type=float, default=0, help="m_fake for ocsoftmax/samo loss")
    parser.add_argument('--alpha', type=float, default=20, help="scale factor for ocsoftmax loss")

    # Other
    parser.add_argument('--continue_training', action='store_true', help="continue training with trained model")
    parser.add_argument('--checkpoint', type=int, help="continue from which epoch")
    parser.add_argument('--test_on_eval', action='store_true',
                        help="whether to run EER on the evaluation set")
    parser.add_argument('--final_test', action='store_true',
                        help="whether to run best model EER on test set")
    parser.add_argument('--test_interval', type=int, default=5, help="test on eval for every how many epochs")
    parser.add_argument('--save_interval', type=int, default=5, help="save checkpoint model for every how many epochs")

    # Test setups
    parser.add_argument('--test_only', action='store_true', help='whether to run test once on chosen model')
    parser.add_argument("--test_model", type=str, default="./models/anti-spoofing_feat_model.pt")
    parser.add_argument("--scoring", type=str, default=None, choices=["fc", "samo", "ocsoftmax"])
    parser.add_argument("--save_score", type=str, default=None,
                        help='score file name to save individual score for each sample')
    parser.add_argument('--save_center', action='store_true', help='whether to save centers as log files')
    parser.add_argument('--dp', action='store_true', default=False, help='use Data Parallel')
    parser.add_argument('--one_hot', action='store_true', help='use one hot vectors in final test')

    # Scenario setups for SAMO
    parser.add_argument('--train_sp', type=int, default=1,
                        help="1: speaker-aware loss(sim score); "
                             "2: speaker-agnostic(maxscore); "
                             "0: other loss")
    # 1: 1 on 1 similarity for all training data
    # 2: maxscore for enrolled training centers
    parser.add_argument('--val_sp', type=int, default=1,
                        help="1: speaker-aware loss(sim score); "
                             "2: speaker-agnostic(maxscore); "
                             "0: speaker-independent(use current train centers)")
    # 0, 1, 2 all based on maxscore with all centers
    # 1 replace score with specific distance for target data, keep maxscore for non-target
    # 0 and 2 differ only for enrollment, 0 enrolls training centers, 2 enrolls test centers
    parser.add_argument('--target', type=int, default=1, help='load target speaker data only in val and eval')
    parser.add_argument('--update_interval', type=int, default=3, help="update training centers for every x epochs")
    parser.add_argument('--init_center', type=int, default=1, help='initialized center with orthogonal embeds')
    parser.add_argument('--update_stop', type=int, default=None, help='stop updating centers after x epochs')
    parser.add_argument("--center_sampler", type=str, default="sequential", choices=["sequential", "random"])

    args = parser.parse_args()

    if not args.dp:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.update_stop == None:
        args.update_stop = args.num_epochs + 1

    # Set seeds
    setup_seed(args.seed)

    if args.continue_training:
        pass
    else:
        # Path for output data
        if os.path.exists(args.out_fold):
            logging.warning('{} exists'.format(args.out_fold))
            print("overwrite:{}".format(args.overwrite))
        if args.out_fold == './models/try/':
            args.overwrite = True
        os.makedirs(args.out_fold, exist_ok=args.overwrite)
        if args.overwrite:
            shutil.rmtree(args.out_fold)
            os.mkdir(args.out_fold)

        # Folder for intermediate results
        if not os.path.exists(os.path.join(args.out_fold, 'checkpoint')):
            os.makedirs(os.path.join(args.out_fold, 'checkpoint'))
        else:
            shutil.rmtree(os.path.join(args.out_fold, 'checkpoint'))
            os.mkdir(os.path.join(args.out_fold, 'checkpoint'))

        # Path for input data
        assert os.path.exists(args.path_to_database)

        # Save training arguments
        with open(os.path.join(args.out_fold, 'args.json'), 'w') as file:
            file.write(json.dumps(vars(args), sort_keys=True, separators=('\n', ':')))
        with open(os.path.join(args.out_fold, 'train_loss.log'), 'w') as file:
            file.write("Start recording training loss ...\n")
        with open(os.path.join(args.out_fold, 'dev_loss.log'), 'w') as file:
            file.write("Start recording validation loss ...\n")
        with open(os.path.join(args.out_fold, 'test_loss.log'), 'w') as file:
            file.write("Start recording test loss ...\n")

    args.cuda = torch.cuda.is_available()
    print('Cuda device available: ', args.cuda)
    args.device = torch.device("cuda" if args.cuda else "cpu")

    return args


def get_loader(args):
    '''
    Make PyTorch DataLoaders for train / developement / evaluation
    Adapted from https://github.com/clovaai/aasist
    '''

    database_path = args.path_to_database
    seed = args.seed
    target_only = args.target

    trn_database_path = database_path + "LA/ASVspoof2019_LA_train/"
    dev_database_path = database_path + "LA/ASVspoof2019_LA_dev/"
    eval_database_path = database_path + "LA/ASVspoof2019_LA_eval/"

    trn_list_path = "protocols/ASVspoof2019.LA.cm.train.trn.txt"
    dev_trial_path = "protocols/ASVspoof2019.LA.cm.dev.trl.txt"
    eval_trial_path = "protocols/ASVspoof2019.LA.cm.eval.trl.txt"

    dev_enroll_path = ["protocols/ASVspoof2019.LA.asv.dev.female.trn.txt",
                       "protocols/ASVspoof2019.LA.asv.dev.male.trn.txt"]
    eval_enroll_path = ["protocols/ASVspoof2019.LA.asv.eval.female.trn.txt",
                        "protocols/ASVspoof2019.LA.asv.eval.male.trn.txt"]

    # Read all training data
    label_trn, file_train, utt2spk_train, tag_train = genSpoof_list(dir_meta=trn_list_path, enroll=False, train=True)
    trn_centers = len(set(utt2spk_train.values()))
    print("no. training files:", len(file_train))
    print("no. training speakers:", trn_centers)
    train_set = ASVspoof2019_speaker_raw(list_IDs=file_train,
                                         labels=label_trn,
                                         utt2spk=utt2spk_train,
                                         base_dir=trn_database_path,
                                         tag_list=tag_train,
                                         train=True)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    # Read bona-fide-only training data
    num_bonafide_train = 2580
    train_set_fix = ASVspoof2019_speaker_raw(list_IDs=file_train,
                                             labels=label_trn,
                                             utt2spk=utt2spk_train,
                                             base_dir=trn_database_path,
                                             tag_list=tag_train,
                                             train=False)
    trn_bona_set = Subset(train_set_fix, range(num_bonafide_train))
    if args.center_sampler == "random":
        trn_bona = DataLoader(trn_bona_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=False,
                              pin_memory=True,
                              worker_init_fn=seed_worker,
                              generator=gen)
    elif args.center_sampler == "sequential":
        trn_bona = DataLoader(trn_bona_set,
                              batch_size=int(args.batch_size),
                              shuffle=False,
                              drop_last=False,
                              pin_memory=True
                              # sampler=torch_sampler.SequentialSampler(range(num_bonafide_train))
                              )

    # Read dev enrollment data
    label_dev_enroll, file_dev_enroll, utt2spk_dev_enroll, tag_dev_enroll = genSpoof_list(dir_meta=dev_enroll_path,
                                                                                          enroll=True,
                                                                                          train=False)
    dev_enroll_spk = set(utt2spk_dev_enroll.values())
    dev_centers = len(dev_enroll_spk)
    print("no. validation enrollment files:", len(file_dev_enroll))
    print("no. validation enrollment speakers:", dev_centers)
    dev_set_enroll = ASVspoof2019_speaker_raw(list_IDs=file_dev_enroll,
                                              labels=label_dev_enroll,
                                              utt2spk=utt2spk_dev_enroll,
                                              base_dir=dev_database_path,
                                              tag_list=tag_dev_enroll,
                                              train=False)
    dev_enroll = DataLoader(dev_set_enroll,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # Read target-only dev data
    label_dev, file_dev, utt2spk_dev, tag_dev = genSpoof_list(dir_meta=dev_trial_path, enroll=False, train=False,
                                                              target_only=target_only, enroll_spk=dev_enroll_spk)
    print("no. validation files:", len(file_dev))
    dev_set = ASVspoof2019_speaker_raw(list_IDs=file_dev,
                                       labels=label_dev,
                                       utt2spk=utt2spk_dev,
                                       base_dir=dev_database_path,
                                       tag_list=tag_dev,
                                       train=False)
    dev_loader = DataLoader(dev_set,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    # Read eval enrollment data
    label_eval_enroll, file_eval_enroll, utt2spk_eval_enroll, tag_eval_enroll = genSpoof_list(dir_meta=eval_enroll_path,
                                                                                              enroll=True,
                                                                                              train=False)
    eval_enroll_spk = set(utt2spk_eval_enroll.values())
    eval_centers = len(eval_enroll_spk)
    print("no. eval enrollment files:", len(file_eval_enroll))
    print("no. eval enrollment speakers:", eval_centers)
    eval_set_enroll = ASVspoof2019_speaker_raw(list_IDs=file_eval_enroll,
                                               labels=label_eval_enroll,
                                               utt2spk=utt2spk_eval_enroll,
                                               base_dir=eval_database_path,
                                               tag_list=tag_eval_enroll,
                                               train=False)
    eval_enroll = DataLoader(eval_set_enroll,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    # Read eval target-only data
    label_eval, file_eval, utt2spk_eval, tag_eval = genSpoof_list(dir_meta=eval_trial_path, enroll=False, train=False,
                                                                  target_only=target_only, enroll_spk=eval_enroll_spk)
    print("no. eval files:", len(file_eval))
    eval_set = ASVspoof2019_speaker_raw(list_IDs=file_eval,
                                        labels=label_eval,
                                        utt2spk=utt2spk_eval,
                                        base_dir=eval_database_path,
                                        tag_list=tag_eval,
                                        train=False)
    eval_loader = DataLoader(eval_set,
                             batch_size=args.batch_size,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    num_centers = [trn_centers, dev_centers, eval_centers]

    return trn_loader, dev_loader, eval_loader, \
           trn_bona, dev_enroll, eval_enroll, num_centers


def train(args):
    torch.set_default_tensor_type(torch.FloatTensor)

    # initialize backbone model AASIST
    with open("aasist/AASIST.conf", "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    module = import_module("aasist.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    feat_model = _model(model_config).to(args.device)
    nb_params = sum([param.view(-1).size()[0] for param in feat_model.parameters()])
    print("no. model params:{}".format(nb_params))

    # Setup for DataParallel
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1 and args.dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        feat_model = nn.DataParallel(feat_model)
        feat_model.to(args.device)

    # load datasets
    trainDataLoader, devDataLoader, evalDataLoader, \
    trainBonaLoader, devEnrollLoader, evalEnrollLoader, num_centers = get_loader(args)

    # load previous models for continue training
    if args.continue_training:
        feat_model = torch.load(
            os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_feat_model_%d.pt' % args.checkpoint)).to(
            args.device)
        loss_model = torch.load(
            os.path.join(args.out_fold, 'checkpoint', 'anti-spoofing_loss_model_%d.pt' % args.checkpoint)).to(
            args.device)

    # optimizer & scheduler setup
    optim_config['base_lr'] = args.lr
    optim_config['lr_min'] = args.lr_min
    optimizer = torch.optim.Adam(feat_model.parameters(),
                                 lr=optim_config['base_lr'],
                                 betas=optim_config['betas'],
                                 weight_decay=optim_config['weight_decay'],
                                 amsgrad=optim_config['amsgrad'])
    total_steps = args.num_epochs * \
                  len(trainDataLoader)
    if "cosine" in args.scheduler:
        if args.scheduler == "cosine2":
            total_steps = 2 * total_steps
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                optim_config['lr_min'] / optim_config['base_lr']))
    elif args.scheduler == 'clr':
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=args.lr_min,
            max_lr=args.lr,
            step_size_up=args.clr_step * len(trainDataLoader),
            mode=args.clr_mode,
            cycle_momentum=False)
    optimizer_swa = SWA(optimizer)
    n_swa_update = 0  # number of snapshots of model to use in SWA

    # loss setup
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.9, 0.1])).to(args.device)
    if args.loss == "ocsoftmax":
        ocsoftmax = OCSoftmax(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha,
                              initialize_centers=args.initialize_centers).to(args.device)
        ocsoftmax.train()
        ocsoftmax_optimizer = torch.optim.SGD(ocsoftmax.parameters(), lr=args.lr)
    elif args.loss == "samo":
        samo = SAMO(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha,
                    num_centers=args.num_centers, initialize_centers=args.initialize_centers).to(args.device)
    monitor_loss = args.loss

    # early_stop setup
    early_stop_cnt = 0
    prev_loss = 1e8
    best_epoch = 0

    # Start training
    for epoch_num in tqdm(range(args.num_epochs)):
        feat_model.train()

        ip1_loader, idx_loader, spk_loader, utt_loader = [], [], [], []
        trainlossDict = defaultdict(list)
        devlossDict = defaultdict(list)
        testlossDict = defaultdict(list)
        if args.scheduler == "exp":
            adjust_learning_rate(args, args.lr, optimizer, epoch_num)
        if args.loss == "ocsoftmax":
            adjust_learning_rate(args, args.lr, ocsoftmax_optimizer, epoch_num)
        print('\nEpoch: %d ' % (epoch_num + 1))

        # Setup SAMO attractors
        if args.train_sp:
            # Initialized centers
            if epoch_num == 0 and args.init_center == 1:  # one-hot initialization
                spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
                tmp_center = torch.eye(args.enc_dim)[:num_centers[0]]
                train_enroll = dict(zip(spklist, tmp_center))
            # Update centers per interval
            elif epoch_num % args.update_interval == 0 and epoch_num < args.update_stop:
                train_enroll = update_embeds(args.device, feat_model, trainBonaLoader)
            # Pass centers to loss
            samo.center = torch.stack(list(train_enroll.values()))

        for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(trainDataLoader)):
            feat = feat.to(args.device)
            labels = labels.to(args.device)

            ## forward
            feats, feat_outputs = feat_model(feat)

            ## loss calculate
            if args.train_sp:
                samoloss, _ = samo(feats, labels, spk=spk, enroll=train_enroll, attractor=args.train_sp)
                feat_loss = samoloss
            elif args.loss == "softmax":
                feat_loss = criterion(feat_outputs, labels)
            elif args.loss == "ocsoftmax":
                ocsoftmaxloss, _ = ocsoftmax(feats, labels)
                feat_loss = ocsoftmaxloss

            ## backward
            if args.loss == "softmax":
                trainlossDict[args.loss].append(feat_loss.item())
                optimizer.zero_grad()
                feat_loss.backward()
                optimizer.step()
            elif args.loss == "ocsoftmax":
                ocsoftmax_optimizer.zero_grad()
                trainlossDict[args.loss].append(ocsoftmaxloss.item())
                optimizer.zero_grad()
                feat_loss.backward()
                optimizer.step()
                ocsoftmax_optimizer.step()
            elif args.loss == "samo":
                trainlossDict[args.loss].append(samoloss.item())
                optimizer.zero_grad()
                feat_loss.backward()
                optimizer.step()

            # adjust lr in feat model optimizer in every iteration
            if args.scheduler != "exp":
                scheduler.step()

            ## record
            ip1_loader.append(feats)
            idx_loader.append((labels))

            with open(os.path.join(args.out_fold, "train_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(i) + "\t" +
                          str(trainlossDict[monitor_loss][-1]) + "\n")

        if args.save_center:
            with open(os.path.join(args.out_fold, "train_enroll.log"), "a") as log:
                log.write(str(epoch_num) + "\t" + str(samo.center.detach().numpy()) + "\n")

        # Val the model
        feat_model.eval()
        with torch.no_grad():
            ip1_loader, idx_loader, spk_loader, score_loader = [], [], [], []

            # SAMO inference centers
            if args.val_sp:
                # Define and update dev centers per epoch
                dev_enroll = update_embeds(args.device, feat_model, devEnrollLoader)
                samo.center = torch.stack(list(dev_enroll.values()))

            for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(devDataLoader)):
                feat = feat.to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                if args.val_sp:  # SAMO inference with enrollment
                    if args.target:  # loss calculation for target-only speakers
                        samoloss, score = samo(feats, labels, spk, dev_enroll, args.val_sp)
                    else:
                        samoloss, score = samo.inference(feats, labels, spk, dev_enroll, args.val_sp)
                    devlossDict[args.loss].append(samoloss.item())
                elif args.loss == "softmax":
                    feat_loss = criterion(feat_outputs, labels)
                    score = feat_outputs[:, 0]
                    devlossDict[args.loss].append(feat_loss.item())
                elif args.loss == "ocsoftmax":
                    ocsoftmaxloss, score = ocsoftmax(feats, labels)
                    devlossDict[args.loss].append(ocsoftmaxloss.item())
                elif args.loss == "samo":
                    # Use training centers when no enrollment(val_sp=0)
                    samoloss, score = samo(feats, labels)
                    devlossDict[args.loss].append(samoloss.item())

                ip1_loader.append(feats)
                idx_loader.append((labels))
                score_loader.append(score)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

            with open(os.path.join(args.out_fold, "dev_loss.log"), "a") as log:
                log.write(str(epoch_num) + "\t" +
                          str(np.nanmean(devlossDict[monitor_loss])) + "\t" +
                          str(eer) + "\n")
            print("Val EER: {}".format(eer))

        if args.save_center:
            with open(os.path.join(args.out_fold, "dev_enroll{}.log".format(epoch_num)), "a") as log:
                log.write(str(epoch_num) + "\t" + str(samo.center.detach().numpy()) + "\n")

        # Test the current model
        feat_model.eval()
        if args.test_on_eval:
            if (epoch_num + 1) % args.test_interval == 0:
                with torch.no_grad():
                    ip1_loader, tag_loader, idx_loader, score_loader = [], [], [], []

                    # SAMO inference centers
                    if args.val_sp:
                        # define and update eval centers
                        eval_enroll = update_embeds(args.device, feat_model, evalEnrollLoader)
                        samo.center = torch.stack(list(eval_enroll.values()))

                    for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(evalDataLoader)):
                        if args.feat == "Raw":
                            feat = feat.to(args.device)
                        else:
                            feat = feat.transpose(2, 3).to(args.device)

                        labels = labels.to(args.device)
                        feats, feat_outputs = feat_model(feat)

                        if args.val_sp:  # for SAMO
                            if args.target:  # loss calculation for target-only speakers
                                samoloss, score = samo(feats, labels, spk, eval_enroll, args.val_sp)
                            else:
                                samoloss, score = samo.inference(feats, labels, spk, eval_enroll, args.val_sp)
                            testlossDict[args.loss].append(samoloss.item())
                        elif args.loss == "softmax":
                            feat_loss = criterion(feat_outputs, labels)
                            score = feat_outputs[:, 0]
                            testlossDict[args.loss].append(feat_loss.item())
                        elif args.loss == "ocsoftmax":
                            ocsoftmaxloss, score = ocsoftmax(feats, labels)
                            testlossDict[args.loss].append(ocsoftmaxloss.item())
                        elif args.loss == "samo" or args.loss == "samo_spoofall":
                            samoloss, score = samo(feats, labels)
                            testlossDict[args.loss].append(samoloss.item())

                        ip1_loader.append(feats)
                        idx_loader.append((labels))
                        score_loader.append(score)

                    scores = torch.cat(score_loader, 0).data.cpu().numpy()
                    labels = torch.cat(idx_loader, 0).data.cpu().numpy()
                    eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

                    with open(os.path.join(args.out_fold, "test_loss.log"), "a") as log:
                        log.write(str(epoch_num) + "\t" + str(np.nanmean(testlossDict[monitor_loss])) + "\t" + str(
                            eer) + "\n")
                    print("Test EER: {}".format(eer))

        # save centers per epoch
        if args.save_center:
            with open(os.path.join(args.out_fold, "eval_enroll{}.log".format(epoch_num)), "a") as log:
                log.write(str(epoch_num) + "\t" + str(samo.center.detach().numpy()) + "\n")

        # save checkpoints
        if (epoch_num + 1) % args.save_interval == 0:
            torch.save(feat_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_feat_model_%d.pt' % (epoch_num + 1)))
            if args.loss == "ocsoftmax":
                loss_model = ocsoftmax
            elif args.loss == "samo":
                loss_model = samo
            elif args.loss == "softmax":
                loss_model = None
            else:
                print("What is your loss? You may encounter error.")
            torch.save(loss_model, os.path.join(args.out_fold, 'checkpoint',
                                                'anti-spoofing_loss_model_%d.pt' % (epoch_num + 1)))

        # save best model by lowest val loss
        valLoss = np.nanmean(devlossDict[monitor_loss])
        if valLoss < prev_loss:
            torch.save(feat_model, os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt'))
            if args.loss == "ocsoftmax":
                loss_model = ocsoftmax
            elif args.loss == "samo" or args.loss == "samo_spoofall" or args.loss == "ocsoftmax_IDEAL" or args.loss == "samo_channel":
                loss_model = samo
            elif args.loss == "softmax":
                loss_model = None
            else:
                print("What is your loss? You may encounter error.")
            torch.save(loss_model, os.path.join(args.out_fold, 'anti-spoofing_loss_model.pt'))

            prev_loss = valLoss
            early_stop_cnt = 0
            best_model_test_eer = eer
            best_epoch = epoch_num
            optimizer_swa.update_swa()
            n_swa_update += 1
        else:
            early_stop_cnt += 1

        if early_stop_cnt == 100:
            with open(os.path.join(args.out_fold, 'args.json'), 'a') as res_file:
                res_file.write('\nTrained Epochs: %d\n' % (epoch_num - 49))
            break

    if args.final_test:
        if args.save_score == None:
            args.save_score = args.out_fold[-8:]
        if args.scoring == None:
            args.scoring = args.loss
            if args.loss == "softmax":
                args.scoring == "fc"
        args.test_model = os.path.join(args.out_fold, 'anti-spoofing_feat_model.pt')
        test(args)

    print("Saving best model in epoch {}\n".format(best_epoch))


def update_embeds(device, enroll_model, loader):
    enroll_emb_dict = {}
    with torch.no_grad():
        for i, (batch_x, _, spk, _, _) in enumerate(tqdm(loader)):  # batch_x = x_input, key = utt_list
            batch_x = batch_x.to(device)
            batch_cm_emb, _ = enroll_model(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()

            for s, cm_emb in zip(spk, batch_cm_emb):
                if s not in enroll_emb_dict:
                    enroll_emb_dict[s] = []

                enroll_emb_dict[s].append(cm_emb)

        for spk in enroll_emb_dict:
            enroll_emb_dict[spk] = Tensor(np.mean(enroll_emb_dict[spk], axis=0))

    return enroll_emb_dict


def test(args):
    torch.set_default_tensor_type(torch.FloatTensor)
    if not os.path.exists('./test_scores'):
        os.makedirs('./test_scores')
    save_path = "./test_scores/{}.txt".format(args.save_score)

    if os.path.exists(save_path):
        print("Calculating on existing score file...\n")
        compute_eer_tdcf(args, save_path)

    else:
        # load models
        if args.test_model[-3:] == "pth":
            with open("aasist/AASIST.conf", "r") as f_json:
                config = json.loads(f_json.read())
            model_config = config["model_config"]
            module = import_module("aasist.{}".format(model_config["architecture"]))
            _model = getattr(module, "Model")
            feat_model = _model(model_config).to(args.device)

            nb_params = sum([param.view(-1).size()[0] for param in feat_model.parameters()])
            print("no. model params:{}".format(nb_params))

            feat_model.load_state_dict(
                torch.load(args.test_model, map_location=args.device))
        else:
            feat_model = torch.load(args.test_model).to(args.device)
        print("Model loaded : {}".format(args.test_model))
        print("Using scoring=", args.scoring, "  testing on val_sp=", args.val_sp, "using target-only=", args.target)
        print("Start evaluation...")
        feat_model.eval()

        # load test data and initialize loss
        # reverse [0,1] label when loading aasist pretrain
        _, _, evalDataLoader, trainBonaLoader, _, evalEnrollLoader, _ = get_loader(args)

        if args.scoring == "ocsoftmax":
            ocsoftmax = OCSoftmax(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha,
                                  initialize_centers=args.initialize_centers).to(args.device)
        elif args.scoring == "samo":
            samo = SAMO(args.enc_dim, m_real=args.m_real, m_fake=args.m_fake, alpha=args.alpha).to(args.device)

        with torch.no_grad():

            ip1_loader, utt_loader, idx_loader, score_loader, spk_loader, tag_loader = [], [], [], [], [], []

            if args.scoring == "samo":
                if args.val_sp:
                    # define and update eval centers
                    eval_enroll = update_embeds(args.device, feat_model, evalEnrollLoader)
                else:  # use training centers without eval enrollment
                    if args.one_hot:
                        spklist = ['LA_00' + str(spk_id) for spk_id in range(79, 99)]
                        tmp_center = torch.eye(args.enc_dim)[:20]
                        eval_enroll = dict(zip(spklist, tmp_center))
                    else:
                        eval_enroll = update_embeds(args.device, feat_model, trainBonaLoader)
                samo.center = torch.stack(list(eval_enroll.values()))

            for i, (feat, labels, spk, utt, tag) in enumerate(tqdm(evalDataLoader)):
                feat = feat.to(args.device)
                labels = labels.to(args.device)
                feats, feat_outputs = feat_model(feat)

                # loss cal
                if args.scoring == "samo":
                    if args.target:  # loss calculation for target-only speakers
                        # val_sp = 0 or 2 calculate all maxscore only
                        # val_sp = 1 calculate 1 on 1 scores
                        _, score = samo(feats, labels, spk, eval_enroll, args.val_sp)
                    else:
                        _, score = samo.inference(feats, labels, spk, eval_enroll, args.val_sp)
                elif args.scoring == "fc":
                    if args.test_model[-3:] == "pth":
                        score = feat_outputs[:, 1]  # pretrained networks with reversed labels
                    else:
                        score = feat_outputs[:, 0]  # samo pretrained
                elif args.scoring == "ocsoftmax":
                    _, score = ocsoftmax(feats, labels)

                ip1_loader.append(feats)
                idx_loader.append((labels))
                score_loader.append(score)
                utt_loader.extend(utt)
                spk_loader.extend(spk)
                tag_loader.extend(tag)

            scores = torch.cat(score_loader, 0).data.cpu().numpy()
            labels = torch.cat(idx_loader, 0).data.cpu().numpy()
            eer = em.compute_eer(scores[labels == 0], scores[labels == 1])[0]

        if args.save_score != None:
            with open(save_path, "w") as fh:  # w as in overwrite mode
                for utt, tag, score, label, spk in zip(utt_loader, tag_loader, scores, labels, spk_loader):
                    fh.write("{} {} {} {} {}\n".format(utt, tag, label, score, spk))
            print("Scores saved to {}".format(save_path))

        print("Test EER: {}".format(eer))

        if args.save_score != None:
            compute_eer_tdcf(args, save_path)


if __name__ == "__main__":
    args = initParams()
    # test on trained embeddings with enrollment
    if args.test_only:
        # generate similarity score and test_eer
        test(args)
    else:
        train(args)

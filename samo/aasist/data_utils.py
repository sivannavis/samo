import numpy as np
import soundfile as sf
from torch import Tensor
from torch.utils.data import Dataset

# modified from https://github.com/clovaai/aasist
___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


# protocol to utt_list[speaker_id] = list of utt
def genSpoof_list(dir_meta, enroll=False, train=True, target_only=False, enroll_spk=None):
    d_meta = {}
    utt_list = []
    tag_list = []
    utt2spk = {}

    if not enroll and train:  # read train
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()

        for line in l_meta:
            spk, key, _, tag, label = line.strip().split(" ")

            if key in utt2spk:
                print("Duplicated utt error", key)

            # utt2spk[key] = int(spk[-4:])
            utt2spk[key] = spk
            tag_list.append(tag)
            utt_list.append(key)
            d_meta[key] = 1 if label != "bonafide" else 0  # bona: 0 spoof: 1
    elif not enroll and not train:  # read dev and eval
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()

        for line in l_meta:
            spk, key, _, tag, label = line.strip().split(" ")
            # spk = int(spk[-4:])

            if not target_only or spk in enroll_spk:  # ensure target only speakers
                if key in utt2spk:
                    print("Duplicated utt error", key)

                utt2spk[key] = spk
                utt_list.append(key)
                tag_list.append(tag)
                d_meta[key] = 1 if label != "bonafide" else 0
    else:  # read in enroll data
        for dir in dir_meta:
            with open(dir, "r") as f:
                l_meta = f.readlines()

            for line in l_meta:
                tmp = line.strip().split(" ")

                spk = tmp[0]
                keys = tmp[1].split(",")

                for key in keys:
                    if key in utt2spk:
                        print("Duplicated utt error", key)

                    # utt2spk[key] = int(spk[-4:])
                    utt2spk[key] = spk
                    utt_list.append(key)
                    d_meta[key] = 0
                    tag_list.append("-")
        # print(utt2spk)

    return d_meta, utt_list, utt2spk, tag_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


# utt_list to (input, label, speaker, utt)
class ASVspoof2019_speaker_raw(Dataset):
    def __init__(self, list_IDs, labels, utt2spk, base_dir, tag_list, train=True, cut=64600):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.utt2spk = utt2spk
        self.tag_list = tag_list
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.train = train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        tag = self.tag_list[index]
        X, _ = sf.read(str(self.base_dir + f"flac/{key}.flac"))
        if self.train:
            X_pad = pad_random(X, self.cut)
        else:  # load for dev and eval
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        spk = self.utt2spk[key]
        return x_inp, y, spk, key, tag
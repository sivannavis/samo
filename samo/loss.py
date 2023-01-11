import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import uniform_hypersphere


class OCSoftmax(nn.Module):
    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0, fix_centers=True, initialize_centers="one_hot"):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        if initialize_centers == "one_hot":
            self.center = nn.Parameter(torch.eye(self.feat_dim)[:1], requires_grad=not fix_centers)
        elif initialize_centers == "random":
            self.center = nn.Parameter(torch.randn(1, self.feat_dim), requires_grad=not fix_centers)
        self.softplus = nn.Softplus()

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = F.normalize(self.center, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        scores = x @ w.transpose(0, 1)
        output_scores = scores.clone()

        scores[labels == 0] = self.m_real - scores[labels == 0]
        scores[labels == 1] = scores[labels == 1] - self.m_fake

        loss = self.softplus(self.alpha * scores).mean()

        return loss, output_scores.squeeze(1)


class SAMO(nn.Module):
    def __init__(self, feat_dim=2, m_real=0.5, m_fake=0.2, alpha=20.0, num_centers=20,
                 initialize_centers="one_hot", addNegEntropy=False):

        super(SAMO, self).__init__()
        self.feat_dim = feat_dim
        self.num_centers = num_centers
        self.m_real = m_real
        self.m_fake = m_fake
        self.alpha = alpha
        if initialize_centers == "one_hot":
            self.center = torch.eye(self.feat_dim)[:self.num_centers]
        elif initialize_centers == "evenly":
            pts = np.array(uniform_hypersphere(self.feat_dim, self.num_centers))
            self.center = torch.from_numpy(pts).float()
        self.softplus = nn.Softplus()
        self.addNegEntropy = addNegEntropy
        if self.addNegEntropy:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x, labels, spk=None, enroll=None, spoofprint=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        # update centers if predefined
        # if spoofprint != 0:
        #     self.center = nn.Parameter(torch.stack(list(enroll.values())), requires_grad=False)
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.center, p=2, dim=1).to(x.device)
        scores = x @ w.transpose(0, 1)
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)
        # spk = spk.numpy()
        # print(enroll)

        if spoofprint == 1:
            # for all target only data, speaker-specific center scores
            tmp_w = torch.stack([enroll[id] for id in spk])
            tmp_w = F.normalize(tmp_w, p=2, dim=1).to(x.device)
            final_scores = torch.sum(x * tmp_w, dim=1).unsqueeze(-1)
            # calculate emb_loss by adjusting scores
            maxscores[labels == 0] = self.m_real - final_scores[labels == 0]
        else:
            # every sample is using maxscore with all centers
            final_scores = maxscores.clone()
            maxscores[labels == 0] = self.m_real - maxscores[labels == 0]

        maxscores[labels == 1] = maxscores[labels == 1] - self.m_fake
        emb_loss = self.softplus(self.alpha * maxscores).mean()

        if self.addNegEntropy:
            scores = self.softmax(scores[labels == 0])
            p = scores.sum(0).view(-1)
            p /= p.sum()

            dist_loss = np.log(w.shape[0]) + (p * p.log()).sum()  # using num_centers

            loss = dist_loss * 1e5 + emb_loss
            # print(dist_loss.item(), emb_loss.item())
        else:
            loss = emb_loss
        # loss = self.softplus(self.alpha * newscores[labels == 0]).mean() + \
        #        self.softplus(self.alpha * newscores[labels == 1]).mean()

        return loss, final_scores.squeeze(1)

    def inference(self, x, labels, spk, enroll, attractor=0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).

            Able to deal with samples without enrollment in scenario args.target=0
        """
        # self.center = nn.Parameter(torch.stack(list(enroll.values())), requires_grad=False)
        x = F.normalize(x, p=2, dim=1)
        w = F.normalize(self.center, p=2, dim=1).to(x.device)
        scores = x @ w.transpose(0, 1)
        maxscores, _ = torch.max(scores, dim=1, keepdim=True)
        # spk = spk.numpy()

        if attractor == 1:
            # modify maxscore if it has a speaker center
            final_scores = maxscores.clone()
            for idx in range(len(spk)):
                if spk[idx] in enroll:
                    tmp_w = F.normalize(enroll[spk[idx]], p=2, dim=0).to(x.device)
                    final_scores[idx] = x[idx] @ tmp_w
            # calculate emb_loss by adjusting scores
            maxscores[labels == 0] = self.m_real - final_scores[labels == 0]
        else:
            # use maxscore for all samples with all centers
            final_scores = maxscores.clone()
            maxscores[labels == 0] = self.m_real - maxscores[labels == 0]

        maxscores[labels == 1] = maxscores[labels == 1] - self.m_fake
        emb_loss = self.softplus(self.alpha * maxscores).mean()

        if self.addNegEntropy:
            scores = self.softmax(scores[labels == 0])
            p = scores.sum(0).view(-1)
            p /= p.sum()

            dist_loss = np.log(w.shape[0]) + (p * p.log()).sum()  # using num_centers

            loss = dist_loss * 1e5 + emb_loss
            # print(dist_loss.item(), emb_loss.item())
        else:
            loss = emb_loss

        return loss, final_scores.squeeze(1)

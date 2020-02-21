import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import shuffle
from torch.distributions.multivariate_normal import MultivariateNormal

class KernelDensityLoss(nn.Module):
  """
  Kernel Density Gaussian loss
  Takes a batch of embeddings and corresponding labels.
  """

  def __init__(self, device, emb_size=64, init_bandwidth=1.0, loss_method='softmax', margin_triplet=1.0, optimize_bandwidth=True):
    super(KernelDensityLoss, self).__init__()
    if optimize_bandwidth:
      self.variance = nn.Parameter(torch.tensor(init_bandwidth))
    else:
      self.variance = torch.tensor(init_bandwidth)
    #self.variances = nn.Parameter(torch.tensor(7 * [init_bandwidth]))
    self.emb_size = emb_size
    self.distributions = []
    self.log_probs = []
    self.digit_indices = []
    self.device = device
    self.num_classes = 7
    self.loss_method = loss_method
    self.margin_triplet = torch.FloatTensor([margin_triplet])
    self.margin_triplet = self.margin_triplet.to(device)
    self.softmax = m = nn.LogSoftmax(dim=0)

    assert self.loss_method in ['softmax', 'contrast', 'triplet', 'softmax_contrast', 'all']

    if self.loss_method == 'softmax':
      self.embed_loss = self.softmax_loss
    elif self.loss_method == 'contrast':
      self.embed_loss = self.contrast_loss
    elif self.loss_method == 'triplet':
      self.embed_loss = self.triplet_loss

  def get_log_prob(self, embedding, index_class, index_utt):
    log_probs = []
    for n in range(len(self.distributions[index_class])):
      if n != index_utt:
        log_probs.append(self.distributions[index_class][n].log_prob(embedding))
    return torch.mean(torch.stack(log_probs))

  def triplet_loss(self, embeddings):    
    negative_probs = []
    positive_probs = []
    for index_pos, arr_pos in enumerate(self.digit_indices):
      len_arr_pos = len(arr_pos)
      for pos_idx in range(len_arr_pos):
        pos_prob = self.log_probs[index_pos,pos_idx,index_pos]
        positive_probs.append(pos_prob)
        # Choose randomly a negative sample which lies inside the margin
        neg_classes = [n for n in range(self.num_classes) if n != index_pos]
        shuffle(neg_classes)
        max_prob = -1e6
        is_semihard = False
        for index_neg in neg_classes:
          arr_neg = list(range(len(self.digit_indices[index_neg])))
          shuffle(arr_neg)
          for neg_idx in arr_neg:
            neg_prob = self.log_probs[index_neg,neg_idx,index_pos]
            if (pos_prob > neg_prob and neg_prob + self.margin_triplet > pos_prob):
              is_semihard = True
              break
            elif neg_prob > max_prob:
              max_prob = neg_prob
        if is_semihard:
          negative_probs.append(neg_prob)
        else:
          negative_probs.append(max_prob)
    ### WARNING ###
    # torch.sum() doesn't work fine. Issue: https://github.com/pytorch/pytorch/issues/5863
    positive_probs = torch.stack(positive_probs)
    negative_probs = torch.stack(negative_probs)
    L = F.relu(negative_probs - positive_probs + self.margin_triplet)
    loss = L.sum()

    return loss

  def softmax_loss(self, embeddings):
    # N spoofing classes, M utterances per class
    N, M, _ = list(self.log_probs.size())

    ### WARNING ###
    # torch.sum() doesn't work fine. Issue: https://github.com/pytorch/pytorch/issues/5863
    L = []
    for j in range(N):
      L_row = []
      for i in range(M):
        L_row.append(-self.softmax(self.log_probs[j,i])[j])
        #L_row.append(-F.log_softmax(self.log_probs[j,i], 0)[j])
      L_row = torch.stack(L_row)
      L.append(L_row)
    L_torch = torch.stack(L)
    return F.relu(L_torch).sum()

  def contrast_loss(self, embeddings):
    # N spoofing classes, M utterances per class
    N, M, _ = list(self.log_probs.size())

    ### WARNING ###
    # torch.sum() doesn't work fine. Issue: https://github.com/pytorch/pytorch/issues/5863
    L = []
    for j in range(N):
      L_row = []
      for i in range(M):
        probs_to_classes = self.log_probs[j,i]
        excl_probs_to_classes = torch.cat((probs_to_classes[:j], probs_to_classes[j+1:]))
        L_row.append(torch.max(excl_probs_to_classes) - self.log_probs[j,i,j])
      L_row = torch.stack(L_row)
      L.append(L_row)
    L_torch = torch.stack(L)

    return F.relu(L_torch).sum()

  def forward(self, embeddings, target, size_average=True):
    classes = np.unique(target)
    self.num_classes = len(classes)
    self.digit_indices = [np.where(target == i)[0] for i in range(self.num_classes)]

    #for variance in self.variances:
    #for i in range(self.num_classes):
    #  torch.clamp(self.variances[i], 1e-6)
  
    self.cov_matrix = self.variance * torch.eye(self.emb_size)
    self.cov_matrix = self.cov_matrix.to(self.device)

    self.distributions = []
    for index_class, arr in enumerate(self.digit_indices):
      self.distributions.append([])
      #variance = self.variances[index_class]
      #cov_matrix = variance * torch.eye(self.emb_size)
      #cov_matrix = cov_matrix.to(self.device)
      for n in arr:
        self.distributions[index_class].append(MultivariateNormal(embeddings[n], self.cov_matrix))
    
    print('Distributions shape')
    print(len(self.distributions))
    
    log_probs = []
    for class_idx, class_indices in enumerate(self.digit_indices):
      probs_row = []
      for utt_idx, utterance in enumerate(class_indices):
        probs_col = []
        for class_centroid in range(self.num_classes):
          if class_centroid == class_idx:
            probs_col.append(self.get_log_prob(embeddings[utterance], class_idx, utt_idx))
          else:
            probs_col.append(self.get_log_prob(embeddings[utterance], class_centroid, -1))
        probs_col = torch.stack(probs_col)
        probs_row.append(probs_col)
      probs_row = torch.stack(probs_row)
      log_probs.append(probs_row)
    self.log_probs = torch.stack(log_probs)
    print('Log probs shape')
    print(self.log_probs.shape)

    if self.loss_method == 'all':
      loss = self.softmax_loss(embeddings) + self.contrast_loss(embeddings) + self.triplet_loss(embeddings)
    elif self.loss_method == 'softmax_contrast':
      loss = self.softmax_loss(embeddings) + self.contrast_loss(embeddings)
    else:
      loss = self.embed_loss(embeddings)

    return loss
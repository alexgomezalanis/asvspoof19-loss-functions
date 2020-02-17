import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LCNN_Dataset

rootPath = os.getcwd()
root_dir = '/home2/alexgomezalanis/la-challenge/flac-files'

def train(args, model, start_epoch, criterion, optimizer, device, model_location):

  train_protocol = 'train_la.csv' if args.is_la else 'train_pa.csv'
  dev_protocol = 'dev_la.csv' if args.is_la else 'dev_pa.csv'
  root_dir = '/home2/alexgomezalanis'

  train_dataset = LCNN_Dataset(
    csv_file='./protocols/' + train_protocol,
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    is_evaluating_la=args.is_la,
    dataset='training',
    num_classes=args.num_classes)

  dev_dataset = LCNN_Dataset(
    csv_file='./protocols/' + dev_protocol,
    root_dir=root_dir,
    n_filts=args.num_filts,
    n_frames=args.num_frames,
    window=args.window_length,
    shift=args.frame_shift,
    is_evaluating_la=args.is_la,
    dataset='development',
    num_classes=args.num_classes)

  #train_sampler = CustomSampler(data_source=train_dataset, shuffle=True)
  #dev_sampler = CustomSampler(data_source=dev_dataset, shuffle=False)

  train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_data_workers)
  dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=args.num_data_workers)

  epoch = best_epoch = start_epoch
  num_epochs_not_improving = 0
  best_dev_loss = float('Inf')

  dev_loss = test_epoch(args, model, device, dev_loader, optimizer, criterion)

  while num_epochs_not_improving < args.epochs:
    print('Epoch: ' + str(epoch))
    print('Bandwidth parameters')
    print(list(criterion.parameters()))
    train_epoch(epoch, args, model, device, train_loader, optimizer, criterion)
    dev_loss = test_epoch(args, model, device, dev_loader, optimizer, criterion)

    state = {
      'epoch': epoch,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'criterion': criterion.state_dict()
    }
    torch.save(state, model_location + '/epoch-' + str(epoch) + '.pt')

    if best_dev_loss > dev_loss:
      best_dev_loss = dev_loss
      num_epochs_not_improving = 0
      best_epoch = epoch
      torch.save(state, model_location + '/best.pt')
    else:
      num_epochs_not_improving += 1
    epoch += 1

  print('The best epoch is: ' + str(best_epoch))

def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion):
  model.train()
  pid = os.getpid()
  for batch_idx, sample in enumerate(data_loader):
    (stft, target, _) = sample
    target = torch.LongTensor(target).to(device)
    stft = stft.to(device)
    optimizer.zero_grad()
    output, embeddings = model.forward(stft)
    if args.loss_method == 'softmax':
      loss = criterion(output, target)
    else:
      loss = criterion(embeddings, target)
    loss.backward()
    optimizer.step()

    if batch_idx % args.log_interval == 0:
      print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        pid, epoch, batch_idx * len(stft), len(data_loader.dataset),
        100. * batch_idx / len(data_loader), loss.item()))
      sys.stdout.flush()

def test_epoch(args, model, device, data_loader, optimizer, criterion):
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for batch_idx, sample in enumerate(data_loader):
      (stft, target, _) = sample
      target = torch.LongTensor(target).to(device)
      stft = stft.to(device)
      optimizer.zero_grad()
      output, embeddings = model.forward(stft)
      if args.loss_method == 'softmax':
        test_loss += criterion(output, target).item() # sum up batch loss
      else:
        test_loss += criterion(embeddings, target)

  test_loss /= len(data_loader.dataset)

  print('\nDevelopment set: Average loss: {:.4f}\n'.format(test_loss))
  sys.stdout.flush()

  return test_loss
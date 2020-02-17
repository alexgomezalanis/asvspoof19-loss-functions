from __future__ import print_function, division
import argparse
import torch
import os
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from model import LCNN
from train import train
from eval import eval
from angular_softmax_loss import AngularPenaltySMLoss
from triplet_loss import TripletLoss
#from kernel_density_loss import KernelDensityLoss
from utils.checkpoint import load_checkpoint, create_directory

# Training settings
parser = argparse.ArgumentParser(description='LCNN ASVspoof 2019')
parser.add_argument('--batch-size', type=int, default=210, metavar='N',
                    help='input batch size for training (default: 14)')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 14)')
parser.add_argument('--epochs', type=int, default=3, metavar='N',
                    help='number of epochs for early stopping (default: 15)')
parser.add_argument('--num-data-workers', type=int, default=7,
                    help='How many processes to load data')
parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--version', type=str, default='v1',
                    help='Version to save the model')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=[1, 4], nargs="*",
                    help='how many training processes to use (default: 5)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--num-filts', type=int, default=256,
                    help='How many filters to compute STFT')
parser.add_argument('--num-frames', type=int, default=400,
                    help='How many frames to compute STFT')
parser.add_argument('--window-length', type=float, default=0.025,
                    help='Window Length to compute STFT (s)')
parser.add_argument('--frame-shift', type=float, default=0.010,
                    help='Frame Shift to compute STFT (s)')
parser.add_argument('--margin-triplet', type=float, default='1.0',
                    help='Prob. margin for triplet loss')
parser.add_argument('--emb-size', type=int, default=64, metavar='N',
                    help='embedding size')
parser.add_argument('--load-epoch', type=int, default=-1,
                    help='Saved epoch to load and start training')
parser.add_argument('--eval-epoch', type=int, default=-1,
                    help='Epoch to load and evaluate')
parser.add_argument('--train', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train the model')
parser.add_argument('--eval', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to extract the xvectors')
parser.add_argument('--is-la', default=True, type=lambda x: (str(x).lower() in ['true', 'yes', '1']),
                    help='Whether to train Logical or Physical Access')
parser.add_argument('--num-classes', type=int, default=7, metavar='N',
                    help='Number of training classes (2, 7, 10)')
parser.add_argument('--loss-method', type=str, default='softmax',
                    help='softmax, angular_softmax_sphereface, angular_softmax_cosface')

rootPath = os.getcwd()
                  
if __name__ == '__main__':
  args = parser.parse_args()
  print(args)

  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  torch.manual_seed(args.seed)

  model = LCNN(args.emb_size, args.num_classes).to(device)

  if args.loss_method == 'softmax':
    criterion = nn.CrossEntropyLoss()
  elif args.loss_method == 'angular_softmax_sphereface':
    criterion = AngularPenaltySMLoss(in_features=args.emb_size, out_features=args.num_classes, device=device, loss_type='sphereface')
  elif args.loss_method == 'angular_softmax_cosface':
    criterion = AngularPenaltySMLoss(in_features=args.emb_size, out_features=args.num_classes, device=device, loss_type='cosface')
  elif args.loss_method == 'triplet_loss':
    criterion = TripletLoss(margin=1.0)

  params = list(model.parameters()) + list(criterion.parameters())
  optimizer = optim.Adam(params, lr=args.lr)

  # Model and xvectors path
  dirSpoof = 'LA' if args.is_la else 'PA'
  dirEmbeddings = 'lcnn_' + args.version + '_loss_' + args.loss_method + '_emb_' + str(args.emb_size) + '_classes_' + str(args.num_classes)
  model_location = os.path.join(rootPath, 'models', dirSpoof, dirEmbeddings)
  create_directory(model_location)

  if (args.load_epoch != -1):
    path_model_location = os.path.join(model_location, 'epoch-' + str(args.load_epoch) + '.pt')
    model, optimizer, criterion, start_epoch = load_checkpoint(model, optimizer, criterion, path_model_location)
  else:
    start_epoch = 0

  if args.train:
    train(
      args=args,
      model=model,
      start_epoch=start_epoch,
      criterion=criterion,
      optimizer=optimizer,
      device=device,
      model_location=model_location
    )
  
  if args.eval:
    if args.eval_epoch != -1:
      path_model_location = os.path.join(model_location, 'epoch-' + str(args.eval_epoch) + '.pt')
    else:
      path_model_location = os.path.join(model_location, 'best.pt')

    model, optimizer, criterion, eval_epoch = load_checkpoint(model, optimizer, criterion, path_model_location)

    DICT_NUM_CLASSES = {
      'LA': { 'training': 7, 'development': 7, 'test': 20 },
      'PA': { 'training': 10, 'development': 10, 'test': 10 }
    }
    DICT_PROTOCOLS = {
      'LA': { 'training': 'train_la.csv', 'development': 'dev_la.csv', 'test': 'eval_la.csv' },
      'PA': { 'training': 'train_pa.csv', 'development': 'dev_pa.csv', 'test': 'eval_pa.csv'}
    }

    embeddings_location = os.path.join(rootPath, 'embeddings_lcnn', dirSpoof, dirEmbeddings)
    softmax_location = os.path.join(rootPath, 'softmax_lcnn', dirSpoof, dirEmbeddings)
    # Create embeddings directories
    create_directory(embeddings_location)
    create_directory(softmax_location)

    db = 'LA' if args.is_la else 'PA'
    db_location = os.path.join(embeddings_location, db)
    create_directory(db_location)
    create_directory(os.path.join(softmax_location, db))

    for db_set in ['training', 'development', 'test']:
      set_location = os.path.join(db_location, db_set)
      create_directory(set_location)
      create_directory(os.path.join(softmax_location, db, db_set))
      num_classes = DICT_NUM_CLASSES[db][db_set]
      for n in range(num_classes):
        class_location = os.path.join(set_location, 'S' + str(n))
        create_directory(class_location)
        create_directory(os.path.join(softmax_location, db, db_set, 'S' + str(n)))

    for db_set in ['training', 'development', 'test']:
      print('Eval embeddings ' + db + ' ' + db_set)
      eval(
        protocol=DICT_PROTOCOLS[db][db_set],
        db=db, 
        db_set=db_set,
        args=args,
        model=model,
        embeddings_location=embeddings_location,
        softmax_location=softmax_location,
        device=device,
        mp=mp)
    

  
  
import numpy as np
import sys
import glob
import os
import matlab.engine

def getEER(scores, labels, sign):
  genuineScores = []
  for i in range(len(labels)):
    if (labels[i] == 0):
      if isinstance(scores[i], np.ndarray):
        genuineScores.append(scores[i].tolist()[0])
      else:
        genuineScores.append(sign * scores[i])
  numClasses = np.unique(labels)
  EER = []
  for n in numClasses:
    if (n != 0):
      spoofScores = []
      for i in range(len(labels)):
        if (labels[i] == n):
          if isinstance(scores[i], np.ndarray):
            spoofScores.append(scores[i].tolist()[0])
          else:
            spoofScores.append(sign * scores[i])
      eer = eng.eer(matlab.double(genuineScores), matlab.double(spoofScores)) * 100
      EER.append(eer)
  return EER

def getSoftmaxScores(kind, num_class):
  X_dev = np.load(root + '/softmax_lcnn/' + dirIvectors + '/' + kind + '/X_' + str(num_class) + '.npy')
  Y = num_class * np.ones((X_dev.shape[0]), dtype=np.int32)
  if (num_class == 0):
    Y_binary = np.zeros((X_dev.shape[0]), dtype=np.int32)
  else:
    Y_binary = np.ones((X_dev.shape[0]), dtype=np.int32)
  return (X_dev, Y, Y_binary)

def getFileNamesDev(num_class):
  fileNames = glob.glob(root + '/softmax_lcnn/' + dirIvectors + '/test/S' + str(num_class) + '/*.npy')
  names = []
  for j in fileNames:
    fileNameParts = j.split('.')[0].split('/')
    fileName = fileNameParts[len(fileNameParts) - 1]
    names.append(fileName)
  return names

def printScores(nameFiles, scores, text_file, target):
  for i in range(len(nameFiles)):
    score = scores[i].tolist()[0] if isinstance(scores[i], np.ndarray) else scores[i]
    text_file.write("%s %s %f \n" % (nameFiles[i], target, score))


dirIvectors = sys.argv[1] # softmax, siamese_softmax
isLA = sys.argv[2] == 'True' # True for LA and False for PA
root = os.getcwd()
rootWrite= '/home2/alexgomezalanis/tDCF_v1/scores'
ASV_SCOREFILE = rootWrite + '/asv_test.txt'

dirSpoof = 'LA' if isLA else 'PA'

eng = matlab.engine.start_matlab()
genpath = eng.genpath('/home2/alexgomezalanis/tDCF_v1/bosaris_toolkit.1.06/bosaris_toolkit')
eng.addpath(genpath)
genpath = eng.genpath('/home2/alexgomezalanis/tDCF_v1')
eng.addpath(genpath)

CM_SCOREFILE = rootWrite + '/' + dirSpoof + '_LCNN_softmax.txt'
text_file = open(CM_SCOREFILE, "w")
z_genuine, Y_genuine, _ = getSoftmaxScores('test', 0)
fileNames = getFileNamesDev(0)
printScores(fileNames, z_genuine, text_file, 'bonafide')
if isLA:
  for i in range(7, 20):
    z_spoof, Y_spoof, _ = getSoftmaxScores('test', i)
    fileNames = getFileNamesDev(i)
    printScores(fileNames, z_spoof, text_file, 'spoof')
    z = np.concatenate((z_genuine, z_spoof))
    y = np.concatenate((Y_genuine, Y_spoof))
    print('S' + str(i) + ': ')
    print(getEER(z, y, 1))
else:
  z_spoof, Y_spoof, _ = getSoftmaxScores('test', 1)
  fileNames = getFileNamesDev(1)
  printScores(fileNames, z_spoof, text_file, 'spoof')
  z = np.concatenate((z_genuine, z_spoof))
  y = np.concatenate((Y_genuine, Y_spoof))
  print(getEER(z, y, 1))
text_file.close()
print("%%% SOFTMAX %%%")
results = eng.evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE)

eng = matlab.engine.start_matlab()
genpath = eng.genpath('/home2/alexgomezalanis/tDCF_v1/bosaris_toolkit.1.06/bosaris_toolkit')
eng.addpath(genpath)
genpath = eng.genpath('/home2/alexgomezalanis/tDCF_v1')
eng.addpath(genpath)

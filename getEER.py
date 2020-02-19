import numpy as np
import sys
import glob
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn import mixture
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

def getIdentityVectors(kind, num_class):
  X_dev = np.load(root + '/embeddings_lcnn/' + dirIvectors + '/' + kind + '/X_' + str(num_class) + '.npy')
  Y = num_class * np.ones((X_dev.shape[0]), dtype=np.int32)
  if (num_class == 0):
    Y_binary = np.zeros((X_dev.shape[0]), dtype=np.int32)
  else:
    Y_binary = np.ones((X_dev.shape[0]), dtype=np.int32)
  return (X_dev, Y, Y_binary)

def getFileNamesDev(num_class):
  fileNames = glob.glob(root + '/embeddings_lcnn/' + dirIvectors + '/test/S' + str(num_class) + '/*.npy')
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

def getLDAScores(X, Y):
  '''LDA SVD with covariance'''
  CM_SCOREFILE = rootWrite + '/LA_LCNN_' + dirIvectors + '_lda.txt'
  clf = LinearDiscriminantAnalysis(solver='svd', store_covariance=True)
  clf.fit(X, Y)
  text_file = open(CM_SCOREFILE, "w")
  z_genuine, Y_genuine, _ = getIdentityVectors('test', 0)
  fileNames = getFileNamesDev(0)
  scores = clf.decision_function(z_genuine)
  printScores(fileNames, scores, text_file, 'bonafide')
  for i in range(7, 20):
    z_spoof, Y_spoof, _ = getIdentityVectors('test', i)
    fileNames = getFileNamesDev(i)
    scores = clf.decision_function(z_spoof)
    printScores(fileNames, scores, text_file, 'spoof')
    z = np.concatenate((z_genuine, z_spoof))
    y = np.concatenate((Y_genuine, Y_spoof))
    scoresEER = clf.decision_function(z)
    print('S' + str(i) + ': ')
    print(getEER(scoresEER, y, 1))
  text_file.close()
  print("%%% LDA %%%")
  results = eng.evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE)

def getSVMScores(X, Y_binary):
  CM_SCOREFILE = rootWrite + '/LA_LCNN_' + dirIvectors + '_svm.txt'
  svm_clf = LinearSVC(C=10, loss="hinge")
  svm_clf.fit(X, Y_binary)
  text_file = open(CM_SCOREFILE, "w")
  z_genuine, _, Y_genuine = getIdentityVectors('test', 0)
  fileNames = getFileNamesDev(0)
  scores = -1 * svm_clf.decision_function(z_genuine)
  printScores(fileNames, scores, text_file, 'bonafide')
  for i in range(7, 20):
    z_spoof, _, Y_spoof = getIdentityVectors('test', i)
    fileNames = getFileNamesDev(i)
    scores = -1 * svm_clf.decision_function(z_spoof)
    printScores(fileNames, scores, text_file, 'spoof')
    z = np.concatenate((z_spoof, z_genuine))
    y = np.concatenate((Y_spoof, Y_genuine))
    scoresEER = -1 * svm_clf.decision_function(z)
    print('S' + str(i) + ': ')
    print(getEER(scoresEER, y, 1))
  text_file.close()
  print("%%% SVM %%%")
  results = eng.evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE)

def getSVMOneScores(X_genuine, X_spoof):
  CM_SCOREFILE = rootWrite + '/LA_LCNN_' + dirIvectors + '_svm_one.txt'
  clf_genuine = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
  clf_genuine.fit(X_genuine)
  clf_spoof = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
  clf_spoof.fit(X_spoof)
  text_file = open(CM_SCOREFILE, "w")
  z_genuine, _, y_genuine = getIdentityVectors('test', 0)
  fileNames = getFileNamesDev(0)
  prob_gen_genuine = clf_genuine.score_samples(z_genuine)
  prob_gen_spoof = clf_spoof.score_samples(z_genuine)
  scores_genuine = prob_gen_genuine - prob_gen_spoof
  printScores(fileNames, scores_genuine, text_file, 'bonafide')
  for i in range(7, 20):
    z_spoof, _, y_spoof = getIdentityVectors('test', i)
    fileNames = getFileNamesDev(i)
    prob_spoof_genuine = clf_genuine.score_samples(z_spoof)
    prob_spoof_spoof = clf_spoof.score_samples(z_spoof)
    scores_spoof = prob_spoof_genuine - prob_spoof_spoof
    printScores(fileNames, scores_spoof, text_file, 'spoof')
    scores = np.concatenate((scores_genuine, scores_spoof))
    y = np.concatenate((y_genuine, y_spoof))
    print('S' + str(i) + ': ')
    print(getEER(scores, y, 1))
  text_file.close()
  print("%%% SVM ONE %%%")
  results = eng.evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE)

def getGMMScores(X_genuine, X_spoof):
  CM_SCOREFILE = rootWrite + '/LA_LCNN_' + dirIvectors + '_gmm.txt'
  text_file = open(CM_SCOREFILE, "w")
  clf_genuine = mixture.GaussianMixture(n_components=1)
  clf_genuine.fit(X_genuine)
  clf_spoof = mixture.GaussianMixture(n_components=1)
  clf_spoof.fit(X_spoof)
  z_genuine, _, y_genuine = getIdentityVectors('test', 0)
  fileNames = getFileNamesDev(0)
  prob_gen_genuine = clf_genuine.score_samples(z_genuine)
  prob_gen_spoof = clf_spoof.score_samples(z_genuine)
  scores_genuine = prob_gen_genuine - prob_gen_spoof
  printScores(fileNames, scores_genuine, text_file, 'bonafide')
  for i in range(7, 20):
    z_spoof, _, y_spoof = getIdentityVectors('test', i)
    fileNames = getFileNamesDev(i)
    prob_spoof_genuine = clf_genuine.score_samples(z_spoof)
    prob_spoof_spoof = clf_spoof.score_samples(z_spoof)
    scores_spoof = prob_spoof_genuine - prob_spoof_spoof
    printScores(fileNames, scores_spoof, text_file, 'spoof')
    scores = np.concatenate((scores_genuine, scores_spoof))
    y = np.concatenate((y_genuine, y_spoof))
    print('S' + str(i) + ': ')
    print(getEER(scores, y, 1))
  text_file.close()
  print("%%% GMM %%%")
  results = eng.evaluate_tDCF_asvspoof19(CM_SCOREFILE, ASV_SCOREFILE)

classifier = sys.argv[1] # lda, svm, svm-one, gmm
dirIvectors = sys.argv[2] # softmax, siamese_softmax
root = os.getcwd()
rootWrite= '/home2/alexgomezalanis/tDCF_v1/scores'
ASV_SCOREFILE = rootWrite + '/asv_test.txt'

X, Y, Y_binary = getIdentityVectors('training', 0)
X_genuine = X
for i in range(1, 7):
  X_class, Y_class, Y_binary_class = getIdentityVectors('training', i)
  X = np.concatenate((X, X_class))
  Y = np.concatenate((Y, Y_class))
  Y_binary = np.concatenate((Y_binary, Y_binary_class))

X_class, Y_class, Y_binary_class = getIdentityVectors('training', 1)
X_spoof = X_class
for i in range(2, 7):
  X_class, Y_class, Y_binary_class = getIdentityVectors('training', i)
  X_spoof = np.concatenate((X_spoof, X_class))

eng = matlab.engine.start_matlab()
genpath = eng.genpath('/home2/alexgomezalanis/tDCF_v1/bosaris_toolkit.1.06/bosaris_toolkit')
eng.addpath(genpath)
genpath = eng.genpath('/home2/alexgomezalanis/tDCF_v1')
eng.addpath(genpath)

print("Classifier")
if (classifier == 'lda'):
  print("LDA")
  getLDAScores(X, Y)
elif (classifier == 'svm'):
  print("SVM")
  getSVMScores(X, Y_binary)
elif (classifier == 'svm-one'):
  print("SVM-ONE")
  getSVMOneScores(X_genuine, X_spoof)
elif (classifier == 'gmm'):
  print("GMM")
  getGMMScores(X_genuine, X_spoof)

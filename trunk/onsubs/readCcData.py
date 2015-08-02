#
# read cuda-convnet data
#
# 10 Oct 2013 by Ulrich Stern
#

import cv2
import numpy as np
import sys, argparse, os

from util import *
from common import *

# - - -

META = "batches.meta"
M_MEAN, M_MIN_MAX = 'data_mean', 'YL_min_max'
HIDDEN = [M_MIN_MAX, M_MEAN, 'YL_page_row_col', 'YL_vn_key']

# - - -

def options():
  dd, bn = '/home/uli/data/cifar-10-py-colmajor/', 6
  p = argparse.ArgumentParser(description='Read cuda-convnet data.')
  p.add_argument('-d', dest='dd', default=dd, metavar='D',
    help='data directory (default: %s)'%dd)
  p.add_argument('-b', dest='bn', type=int, default=bn, metavar='N',
    help='batch number (1..6; default: %d)'%bn)
  p.add_argument('--imgs', dest='imgs', metavar='L',
    help='show the given list of images (comma-separated)')
  p.add_argument('-m', '--mean', dest='mean', action='store_true',
    help='calculate and show data_mean')
  p.add_argument('-a', '--all', dest='all', action='store_true',
    help='show also %s entries that are hidden by default  (%s)' \
      %(META, ', '.join(HIDDEN)))
  return p.parse_args()

# - - -

def readFile(fn, verbose=True):
  if verbose:
    print "\n=== %s ===" %fn
  d = unpickle(checkIsfile(opts.dd, fn))
  if verbose:
    print "keys:", d.keys()
  return d

def printValInfo(d, key):
  val = d.get(key)
  print key+":",
  if isinstance(val, list):
    print len(val), type(val[0]) if val else '-'
  elif isinstance(val, np.ndarray):
    print val.shape, val.dtype
  else:
    print val
  return val

# - - -
  
# read data_batch and show some images
def readBatch():
  d = readFile(dbFn(opts.bn))

  data = printValInfo(d, 'data')
  labels = printValInfo(d, 'labels')

  # determine image shape
  imgSz, nCh = DataBatch.imgSizeNumChannels(data)
  print "images: size %d, channels %d" %(imgSz, nCh)

  imgIdxs = map(int, opts.imgs.split(',')) if opts.imgs else []
  if imgIdxs:
    lns = readFile(META, verbose=False)['label_names']
  for i in imgIdxs:
    img = DataBatch.getImageFromData(data, i, imgSz, nCh)
    wn = "batch %d, image %d, label '%s'" %(opts.bn, i, lns[labels[i]])
    cv2.namedWindow(wn, 0)   # makes image larger
    cv2.imshow(wn, img)
  cv2.waitKey(0)

# read batches.meta
def readMeta():
  d = readFile(META)

  printValInfo(d, M_MEAN)
  printValInfo(d, M_MIN_MAX)
  if not opts.all:
    if M_MIN_MAX in d:
      print "first 5 elements of each sublist of %s:" %(M_MIN_MAX)
      for idx, mmB in enumerate(d[M_MIN_MAX]):
        print "  %d: %s" %(idx, mmB[0:5])
    for k in HIDDEN:
      if k in d and not (opts.mean and k == M_MEAN):
        d[k] = '__hidden__'
  print d

# determine data_mean from batches
def dataMean():
  maxB = 5   # maximum training batch number (1..5)
  print "\n=== data_mean batch 1..%d ===" %maxB
  batchMeans, batchSizes = [], np.zeros((maxB,), np.int)
  for bn in range(1,maxB+1):
    data = readFile(dbFn(bn), verbose=False)['data']
    batchMeans.append(np.mean(data, axis=1))
    batchSizes[bn-1] = data.shape[1]
  if not all(batchSizes == batchSizes[0]):
    print "batch sizes differ:", batchSizes
    return
  dm = np.mean(batchMeans, axis=0).astype(np.float32).reshape(-1, 1)
  print dm

# - - -

opts = options()
print "dir: %s" %opts.dd
readBatch()
readMeta()
if opts.mean:
  dataMean()


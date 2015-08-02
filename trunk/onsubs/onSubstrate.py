#
# human "on substrate" classification to create data for cuda-convnet
#
# 7 Sep 2013 by Ulrich Stern
#
# notes:
# * environment variable YL_USER used to select user (default: uli)
#

import cv2
import numpy as np, scipy.io
import argparse, os, sys, time, atexit, shutil
import random, collections, copy
from operator import itemgetter
from lockfile import LockFile
import psutil, pylru

from util import *
from common import *

# - - -

DEBUG, VERBOSE = False, False

LOG_FILE, CREATION_LOG = "__onSubstrate.log", "__creation.log"

FLY_VIDEOS = FlyVideo.FNS[:8]   # subset for human classification

SHARED_DATA_DIR = DATA_DIR+'class/'
SHARED_DATA_LOCK = SHARED_DATA_DIR+'__lock'
PREDS_SUBDIR = 'preds/'

# fly images page-related
N_ROWS, N_COLS, IMG_DIST =  5, 10, 10
TXT_STL_DIFF = (cv2.FONT_HERSHEY_PLAIN, 1, COL_Y, 1, cv2.CV_AA)
TXT_STL_HDR = (cv2.FONT_HERSHEY_PLAIN, 1.1, COL_BK, 1, cv2.CV_AA)
COL_BG_NEW = (0x7c, 0xe6, 0xdf)

# warn if fly image extends into border by this or more
WARN_BORDER_X = 2

ALL_VIDEOS = 'all'
CORR_TH, N_BEFORE_AFTER, NUM_EVAL_IMGS = 0.95, 3, 2500

# clustering
MAX_DIST = 2   # distance from previous classification
TXT_STL_HDR_CL = (cv2.FONT_HERSHEY_PLAIN, 0.9, COL_BK, 1, cv2.CV_AA)
TXT_STL_ON_CL = (cv2.FONT_HERSHEY_PLAIN, 0.8, COL_G, 1, cv2.CV_AA)

PRED_DATA_DIR = DATA_DIR+'cc-exp/for-pred/'
PRED_DATA_BATCH = DataBatch(PRED_DATA_DIR, [])
CC_DIR = ccDir(dropOut=False)
MODEL_DIR = EXP_DIR+'2014-02-02__20-20-47/ConvNet__2014-02-02_20.20.48/'

# - - -

# globals
preds = None
hc = anonObj()   # human classification

# - - -

def options():
  videoNum, flyImgSz, intrsctnSize = 0, 64, 32
  ccDataDir, batchSize, showClusterTh = DATA_DIR+'cc-exp/', 400, 10
  nvids = len(FLY_VIDEOS)
  p = argparse.ArgumentParser(
    description='Human "on substrate" classification.')
  p.add_argument('-v', dest='videoNum', type=int, default=videoNum,
    metavar='N', choices=xrange(nvids),
    help='video number (0-%d' %(nvids-1) + ', see --vl, default: %(default)s)')
  p.add_argument('--fis', dest='fis', type=int, default=flyImgSz,
    metavar='N', help='fly image size in pixels (default: %(default)s)')
  p.add_argument('--sharedHc', dest='sharedHc', action='store_true',
    help='use "shared" human classification file (by default, ' +
      'classifications from the check-in directory are used for ' +
      'cuda-convnet data and --hcs)')

  g = p.add_argument_group('human classification (default)')
  g.add_argument('--diff', dest='showDiffs', action='store_true',
    help='show classification and tagging differences compared to other users')
  g.add_argument('--ell', dest='showEllipses', action='store_true',
    help='show Ctrax ellipses on fly image pages')
  g.add_argument('--intrs', dest='showPatchIntrsctn', action='store_true',
    help='show intersection of random patches used by cuda-convnet')
  g.add_argument('--is', dest='intrsctnSize', type=int,
    default=intrsctnSize, metavar='N',
    help='intersection size (default: %(default)s); calculation: ' +
      '(fly image size) - 4 * (crop border)')
  g.add_argument('--bMax', dest='bMax', type=float, metavar='V',
    nargs='?', const=-1, default=None,
    help='print bMax and abMin from "on substrate" MAT-file (if available)' +
      ' and possibly adjust bMax')
  g.add_argument('--ign', dest='ignoreStored', action='store_true',
    help='ignore previously stored classification')

  g = p.add_argument_group('analyze human classifications')
  g.add_argument('--ana', dest='analyze', action='store_true',
    help='analyze clusters')
  g.add_argument('--cs', dest='showClusterTh', type=int,
    default=showClusterTh, metavar='N',
    help='show images of clusters of at least this size ' +
      '(default: %(default)s)')
  g.add_argument('--susp', dest='showSuspClusters', action='store_true',
    help='show images of clusters with change in classification or ' +
      'corr >= %.2f' %CORR_TH)
  g.add_argument('--pgs', dest='showPageNums', action='store_true',
    help='show page numbers of the cluster images')
  g.add_argument('--ignP', dest='ignorePredictions', action='store_true',
    help='ignore predictions (e.g., to see clusters for initial image ' +
      'choice method)')

  g = p.add_argument_group('cuda-convnet data')
  meg = g.add_mutually_exclusive_group()
  meg.add_argument('-w', dest='writeCcData', action='store_true',
    help='write cuda-convnet data')
  meg.add_argument('--info', dest='ccDataInfo', action='store_true',
    help='show same info as when writing cuda-convnet data but do not write')
  meg.add_argument('--preds', dest='predict', action='store_true',
    help='use cuda-convnet to generate predictions (for --nb batches)')
  meg.add_argument('--chkPreds', dest='checkPreds', action='store_true',
    help='check whether method 1 (see --eval) was used for predictions')
  g.add_argument('-d', dest='ccDataDir', default=ccDataDir, metavar='D',
    help='directory for writing cuda-convnet data (default: %(default)s)')
  g.add_argument('-b', dest='batchSize', type=int, default=batchSize,
    metavar='N', help='batch size (default: %(default)s)')
  g.add_argument('--nb', dest='numBatches', type=int, default=None,
    metavar='N', help='number of batches (per video, default: calculated ' +
      'based on batch size and number of good classifications, or 6 for ' +
      'predictions)')
  g.add_argument('--vids', dest='videos', metavar='L',
    nargs='?', const=ALL_VIDEOS, default=None,
    help='use all or the given list of videos (comma-separated)')
  g.add_argument('--src', dest='ccDataDirSrc', default=None, metavar='D',
    help='create cuda-convnet data by copying this directory and then ' +
      'replacing data_batch_1,2,... with "test" batches')

  g = p.add_argument_group('other commands')
  g.add_argument('--vl', dest='videoList', action='store_true',
    help='show video list')
  g.add_argument('--hcs', dest='hcStats', action='store_true',
    help='show human classification stats for all videos')
  g.add_argument('--hcd', dest='hcDiff', action='store_true',
    help='show differences in human classification between files in ' +
      'shared and checked-in directories for all videos')
  g.add_argument('--cpPreds', dest='copyPreds', action='store_true',
    help='copy prediction files (to %s)' %PREDS_SUBDIR)
  g.add_argument('--eval', dest='evalImageChoice', type=int, metavar='M',
    nargs='?', const=0, default=None, choices=range(3),
    help='evaluate image choice method (0: no restriction, ' +
      '1: %d before/after' %N_BEFORE_AFTER +
      ' corrs < %.2f, 2: all corrs < %.2f;' %(CORR_TH, CORR_TH) +
      ' default: %(const)s)')
  g.add_argument('--play', dest='playVideo', action='store_true',
    help='play video')
  g.add_argument('--users', dest='classifyingUsers', action='store_true',
    help='show classifying users')
  return p.parse_args()

def overrideOptions():
  ov = []
  if opts.predict:
    fis, bSz = PRED_DATA_BATCH.imgSize(), PRED_DATA_BATCH.batchSize()
    if opts.fis != fis:
      ov.append(("fly image size", fis))
      opts.fis = fis
    if opts.batchSize != bSz:
      ov.append(("batch size", bSz))
      opts.batchSize = bSz
    if opts.ccDataDir != PRED_DATA_DIR:
      ov.append(("cuda-convnet data dir", PRED_DATA_DIR))
      opts.ccDataDir = PRED_DATA_DIR
  if ov:
    print "overriding\n  %s\n" %"\n  ".join("%s to %s" %e for e in ov)

# - - - local utils - - -

# returns the fly image size/2 for correlation
def fisHCorr():
  fis = {92:48, 64:48}.get(opts.fis)
  assert fis
  return fis/2

# returns the next (fly, frame index) tuple for which the fly image has
#  normalized correlation smaller than CORR_TH
# note: no longer used
def nextAllowedImage(key, fisH, cap, img=None):
  if img is None:
    img = getFlyImage(key, fisH, cap=cap)[0]
  f, fi = key
  while True:
    fi += 1
    if fi >= x.shape[1]:
      return (1, 0) if f == 0 else None
    img2 = getFlyImage((f, fi), fisH, cap=cap)[0]
    if normCorr(img, img2) < CORR_TH:
      return f, fi

# returns the predictions filename
def predsFn(videofn):
  return FlyVideo.changeExt(videofn, ' preds')

# - - -

# loads predictions file and sets preds
def loadPreds(videofn):
  global preds
  pfn = predsFn(videofn)
  if opts.ignorePredictions or not os.path.exists(pfn):
    preds = None
  else:
    preds = unpickle(pfn)['pred2keys']
    print "loaded predictions file (predictions: %s)" %", ".join(
      "%s:%d" %(lbl, len(preds[i])) for i, lbl in enumerate(LABEL_NAMES))

# - - - human classification - - -

# show classification image (fly images page)
def cimshow(setCallback=False):
  winName = 'human classification'
  img = hc.img.copy()
  for idx, onSubs in enumerate(hc.onSubs):
    x, y = hc.tl[idx]
    x2, y2 = tupleAdd((x, y), opts.fis)
    cv2.rectangle(img, (x, y), (x2, y2), COL_G if onSubs else COL_R, 1)
    putText(img, 'on' if onSubs else 'off', (x+2, y+2), (0, 1),
      TXT_STL_DIFF if opts.showDiffs and hc.onSubsDiff[idx] else
        (TXT_STL_ON if onSubs else TXT_STL_OFF))
    if hc.tagged[idx]:
      putText(img, 'x', (x2-2, y2-3), (-1, 0), TXT_STL_TAG)
    if opts.showDiffs and hc.taggedDiff[idx]:
      putText(img, '+', (x+2, y2-3), (0, 0), TXT_STL_TAG)
    if opts.showPatchIntrsctn:
      d = (opts.fis - opts.intrsctnSize) / 2
      cv2.rectangle(img, tupleAdd((x, y), d), tupleSub((x2, y2), d), COL_W, 1)
  cv2.imshow(winName, img)
  if setCallback:
    cv2.setMouseCallback(winName, onmouse)

# handle mouse click
def mouseClick(pos, left=True):
  if pos:
    x, y = pos
    idx = hc.idxImg[y, x]
    if idx >= 0:
      if left:
        hc.onSubs[idx] = not hc.onSubs[idx]
      else:
        hc.tagged[idx] = not hc.tagged[idx]
      if opts.showDiffs:
        updateDiffs()
      cimshow()
      hc.unchanged = False

# mouse callback
def onmouse(event, x, y, flags, param):
  if event == cv2.EVENT_LBUTTONDOWN:
    mouseClick((x, y))
  elif event == cv2.EVENT_RBUTTONDOWN:
    mouseClick((x, y), False)

# update classification or tagging differences compared to other users
def updateDiffs(setOtherUsers=False):
  if setOtherUsers:
    hc.otherUsers = \
      [k for k in hc.pg.keys() if k.startswith('_') and k != hc.user]
  onSubsDiff = taggedDiff = len(hc.onSubs)*[False]
  onSubs, tagged = [np.array(a) for a in [hc.onSubs, hc.tagged]]
  for ou in hc.otherUsers:
    onSubsDiff = np.logical_or(onSubsDiff, onSubs != hc.pg[ou]['onSubs'])
    taggedDiff = np.logical_or(taggedDiff, tagged != hc.pg[ou]['tagged'])
  hc.onSubsDiff, hc.taggedDiff = onSubsDiff, taggedDiff

# show fly images for the given page
def showFlyImages(fv, pgid, newPg):
  nr, nc, d, fisH = N_ROWS, N_COLS, IMG_DIST, opts.fis/2
  nfis = nr*nc
  pg = hc.d[pgid]
  if not hc.user in pg:
    pg[hc.user] = dict(onSubs=nfis*[pg['onSubsPage']], tagged=nfis*[False])
    hc.unchanged = False
  time, flies, frames = [pg[k] for k in ['time', 'flies', 'frames']]
  onSubs, tagged = [pg[hc.user][k] for k in ['onSubs', 'tagged']]

  npgs = len(hc.d)
  hdrP = 'new page (%d)' %pgid if newPg else 'page %d of %d' %(pgid, npgs)
  hdr = 'v%d  %s  (generated %s)  user: %s' \
    %(fv.videoNum(), hdrP, time2str(time), hc.user[1:])
  hdrH = textSize(hdr, TXT_STL_HDR)[1]
  hL, wL = nr*(opts.fis+d)+hdrH+d, nc*(opts.fis+d)
  imgL = getImg(hL, wL)
  if newPg:
    imgL[0:d+hdrH,:] = COL_BG_NEW
  cv2.putText(imgL, hdr, (d/2, d/2+hdrH), *TXT_STL_HDR)

  setIdxImg = not hasattr(hc, 'idxImg')
  if setIdxImg:
    idxImg = np.zeros((hL, wL), np.int16)
    idxImg[:,:] = -1
    tl = nfis*[None]
  frms = fv.getFrames(frames)
  for r in range(nr):
    for c in range(nc):
      idx = nc*r + c
      if idx < len(frames):
        f, fi = flies[idx], frames[idx]
        img = frms[fi]
        if opts.showEllipses:
          fv.ellipse(img, f, fi)
        fimg, bx = fv.getFlyImage((f, fi), fisH, frms=frms)
        xL, yL = d/2+c*(opts.fis+d), hdrH+d + d/2+r*(opts.fis+d)
        if opts.showPatchIntrsctn and bx >= WARN_BORDER_X:
          b = min(d/2, 4)
          imgL[yL-b:yL+opts.fis+b+1, xL-b:xL+opts.fis+b+1] = COL_BK
        imgL[yL:yL+opts.fis, xL:xL+opts.fis] = fimg
        if setIdxImg:
          idxImg[yL:yL+opts.fis, xL:xL+opts.fis] = idx
          tl[idx] = (xL, yL)
  if setIdxImg:
    hc.idxImg = idxImg
    hc.tl = tl
  hc.img = imgL
  hc.onSubs, hc.tagged = onSubs, tagged
  if opts.showDiffs:
    hc.pg = pg
    updateDiffs(True)
  cimshow(True)

# human classification
def humanClassification(fv):
  printInstructions()
  hc.unchanged = True
  if not hc.d:
    pgid, newPg = addPage(fv)
  else:
    pgids = [pgid for pgid, pg in hc.d.iteritems() if hc.user in pg]
    pgid, newPg = max(pgids) if pgids else 1, False
  shownPgid = -1
  while True:
    if shownPgid != pgid:
      showFlyImages(fv, pgid, newPg)
      shownPgid, newPg = pgid, False
    c = chr(cv2.waitKey(0) & 255)
    if c == 'n':
      pgid, newPg = addPage(fv) if pgid == len(hc.d) else (pgid+1, False)
    elif c == 'p':
      pgid = max(1, pgid-1)
    elif c == 'f':
      pgid = 1
    elif c == 'l':
      pgid = len(hc.d)
    elif c == 'q':
      if hc.unchanged:
        print "no changes"
        break
      print "save ('y'/'n'/'c'=cancel quit)?"
      c = chr(cv2.waitKey(0) & 255)
      if c == 'y':
        saveHumanClassification()
        break
      elif c == 'n':
        break

def printInstructions():
  print "--- human classification ---"
  if opts.showDiffs:
    print 'showing differences vs. other users:'
    print ' classification: yellow "on"/"off", tagging: +'
  if opts.showPatchIntrsctn:
    print 'images that extend >= %d into frame border have black border' \
      %WARN_BORDER_X
  print """
mouse and keyboard commands:
(note: focus must be on the image window for keyboard commands to work)

toggle "on substrate": left click
toggle "tagged":       right click
next page:             'n'  (note: adds page if on last page)
previous page:         'p'
first/last page:       'f'/'l'
quit:                  'q'  (note: asks whether to save)
"""

# - - -

# note: see --eval for the definition of the image choice methods used in
#  the following functions

def getGetImg(fv, fisH=None):
  fisH = fisH or fisHCorr()
  def getImg(key): return fv.getFlyImage(key, fisH)[0]
  return getImg

# returns whether key good according to method 1
def isGood(key, keys, getImg):
  corrKeys, img = beforeAfter(keys, key, N_BEFORE_AFTER), getImg(key)
  return all(normCorr(getImg(k), img) < CORR_TH for k in corrKeys if k)

# check correlations of the predictions using method 1
# note: calculating correlations in one direction would be sufficient
def checkPreds(fv):
  print "checking whether method 1 (see --eval) was used for predictions..."
  keys = blist.sortedlist(set.union(*preds.values()))
  getImg = getGetImg(fv)
  nng = sum(not isGood(key, keys, getImg) for key in keys)
  print "  %d predictions, %s good" %(
    len(keys), "all" if nng == 0 else "%d not" %nng)

# chooses classifications using method 1
def chooseClassifications(fv, cls):
  print "\nchoosing among %d classifications using method 1" %len(cls)
  cands, keys = cls.keys(), blist.sortedlist()
  getImg = getGetImg(fv)
  random.shuffle(cands)
  for key in cands:
    if isGood(key, keys, getImg):
      keys.add(key)
  print "  got %d good ones" %len(keys)
  return keys

# chooses on/off candidate images, possibly with low correlation
def chooseImages(fv, numImgs, keys=None, mthd=1, forEval=False):
  print "\nchoosing %d%s images using method %d%s" %(numImgs,
    "-%d" %len(keys) if keys else "", mthd,
    " (with %d before/after)" %N_BEFORE_AFTER if mthd == 1 else "")

  ks = set(keys) if keys else set()
  cands = dict((onS, fv.inSapKeys(onS) - ks) for onS in [True, False])
  def nCands():
    return [len(cands[onS]) for onS in [True, False]]
  print "number of image candidates [on, off]: %s" %nCands()

  getImg = getGetImg(fv)
  if keys:
    assert isinstance(keys, blist.sortedlist)
    imgs, cg = dict((k, getImg(k)) for k in keys), len(keys)
    assert cg <= numImgs
  else:
    imgs, keys, cg = {}, blist.sortedlist(), 0
  onS, good, cb, t = True, True, 0, Timer()
  while cg < numImgs:
    key = random.sample(cands[onS], 1)[0]
    cands[onS].remove(key)
    img = getImg(key)

    if mthd > 0:
      corrKeys = beforeAfter(keys, key, N_BEFORE_AFTER) if mthd == 1 else \
        imgs.keys()
      good = all(normCorr(imgs[key], img) < CORR_TH for key in corrKeys if key)

    if good:
      imgs[key] = img
      keys.add(key)
      cg += 1
      if mthd == 2 and numImgs > 1000:
        f, fi = key
        for d in [1, -1]:
          rem = set()
          while True:
            fi += d
            if fi < 0 or fi >= fv.tlen(): break
            if normCorr(img, getImg((f, fi))) < CORR_TH:
              break
            rem.add((f, fi))
          cands[onS] -= rem
      if cg % 100 == 0:
        print "  %d (bad: %d, cands: %s)  [%.1fs]" %(cg, cb, nCands(), t.get())
      onS = not onS
    else:
      cb += 1

  if forEval:
    print "\ngood: %d, bad: %d" %(cg, cb)

  return keys, imgs

# evaluate image choice method
def evalImageChoice(fv):
  keys, imgs = chooseImages(fv, NUM_EVAL_IMGS, mthd=opts.evalImageChoice,
    forEval=True)

  print "calculating pairwise correlations...\n ",
  tc = nc = nlc = 0
  for i in range(len(keys)):
    if (i+1) % 100 == 0:
      print (i+1),
      sys.stdout.flush()
    for j in range(i+1, len(keys)):
      corr = normCorr(imgs[keys[i]], imgs[keys[j]])
      tc += corr
      nc += 1
      nlc += corr >= CORR_TH
      
  print "\naverage: {:.1%}, >= {:.2f}: {:.3%}".format(
    tc/nc, CORR_TH, float(nlc)/nc)

# - - -

# plays the video
def playVideo(fv):
  tlen = fv.tlen()
  print "trajectory lengths: %d" %(tlen)

  fi = 0
  while True:
    ret, img = fv.read()
    if not ret:
      break
    if fi < tlen:
      for f in [0,1]:
        fv.ellipse(img, f, fi, cols=[COL_G, COL_R])
    cv2.imshow("video", img)
    if fi % 10 == 0:
      cv2.waitKey(1)
    fi += 1

  print "number of frames: %d" %(fi)

# - - -

# human classification file:
# dictionary
# * key: page id (1, 2, ...)
# * value: dictionary with
#    time:   when page of fly images generated in seconds since epoch
#    onSubsPage: whether the machine assumed the images on the page are "on
#                 substrate"
#    flies:  array with fly number for each fly image
#    frames: array with frame for each fly image
#    _<user>:  dictionary with
#      onSubs: array with "on substrate" classification for each fly image
#      tagged: array with "tagged" status for each fly image

def getUser(): return os.environ.get('YL_USER') or 'uli'

def hcErrorsMsg(msg, n):
  print "  %s %d human classification error%s" %(msg, n, "" if n==1 else "s")

# handle human classification errors
def hcErrors(vn):
  hc.errs, fn = readHcErrors()
  if os.path.isfile(fn):
    print "  using revision %s of %s" %(svnRevision(fn)[0], HC_ERROR_FILE)
    if not (opts.writeCcData or opts.ccDataInfo):   # tag
      nt, user = 0, hc.user
      for vnE, (pg, r, c) in hc.errs:
        if vnE == vn:
          hc.d[pg][user]['tagged'][r*N_COLS+c] = True
          nt += 1
      hcErrorsMsg("tagged", nt)

# returns whether human classification file exists
def loadHumanClassification(vn, svnHc=True):
  vidFn = FLY_VIDEOS[vn]
  fn = FlyVideo.changeExt(vidFn, '_class.data')
  hcbn = os.path.basename(fn)
  def hcfn():
    return (SVN_DATA_DIR if svnHc else SHARED_DATA_DIR) + hcbn
  fn = hcfn()
  if svnHc and not os.path.isfile(fn):
    svnHc = False
    fn = hcfn()
  exists = os.path.isfile(fn)
  if exists:
    print "using %s human classification file" \
      %("revision %s of" %svnRevision(fn)[0] if svnHc else '"shared"')
  else:
    print "no human classification file"
  hc.fn = fn
  hc.d = {} if opts.ignoreStored else unpickle(fn) or {}
  hc.user = '_' + getUser()
  if exists:
    hcErrors(vn)
  return exists

def saveHumanClassification():
  pickle(hc.d, hc.fn, backup=True)

# returns None if lock was acquired and user holding lock otherwise
@atexit.register
def hcLock(acquire=False, verbose=False):
  if not 'opts' in globals() or opts.videos:
    return
  with LockFile(SHARED_DATA_LOCK):
    vn, userPid = opts.videoNum, (getUser(), os.getpid())
    d = unpickle(SHARED_DATA_LOCK) or {}   # entries: vn -> (user, pid)
    if acquire and verbose:
      print "acquiring lock for %s (directory pre: %s)" %(userPid, d)
    locker = d.get(vn)
    if acquire:
      pids = psutil.get_pid_list()
      assert userPid[1] in pids
      if locker is not None and locker[1] in pids:
        return locker[0]
      d[vn] = userPid
    else:
      if locker == userPid:
        del d[vn]
    pickle(d, SHARED_DATA_LOCK)
    return None

# returns map from image key to (page, row, column)
def key2page():
  k2p = {}
  for pgid, pg in hc.d.iteritems():
    for i, key in enumerate(zip(pg['flies'], pg['frames'])):
      r, c = divmod(i, N_COLS)
      k2p[key] = (pgid, r, c)
  return k2p

def completedByUser():
  u2pgid = {}
  for pgid, pg in sorted(hc.d.iteritems()):
    for user in users(pg, underscore=False):
      u2pgid[user] = pgid
  return u2pgid

# human classification stats
def hcStats(allPages=True):
  for first in [True, False]:
    c = goodClassifications(allPages=allPages)[1]
    if c['pages'] > 0:
      if first:
        print "pages completed by user: %s" %", ".join(
          "%s:%d" %kv for kv in sorted(completedByUser().iteritems()))
      elif nUsers == c['users']:
        break
      na, nt, nd, ngnp, nUsers = \
        [c[k] for k in ['all', 'tagged', 'diff', 'goodNotInPreds', 'users']]
      print 'images with at least ' + \
        '%d human classification%s (number pages: %d):' \
        %(nUsers, 's' if nUsers > 1 else '', c['pages'])
      print ' tagged:                      %d' %nt
      print ' different but not tagged:    %d' %nd
      print ' good but not in predictions: %d' %ngnp
      print ' good:                        %d' %(na - (nt+nd+ngnp))
      if opts.hcStats:
        print ' mean human error rate     >= {:.2%}'.format(float(nd)/3/(na-nt))
    if allPages:
      allPages = False
    else:
      break

# update (or create) human classification map
#  key: (fly, frame), value: onSubs
def updateHcMap():
  hc.m = {}
  for pgid, pg in hc.d.iteritems():
    for f, fi, onS in zip(pg['flies'], pg['frames'], pg[hc.user]['onSubs']):
      hc.m[(f, fi)] = onS

# add page, returning page id and whether page added
def addPage(fv):
  ids = hc.d.keys()
  pgid = max(ids) + 1 if ids else 1
  pg = dict(time=int(time.time()))

  # determine whether to try to pick onSubs or not
  updateHcMap()
  nHc, nOnS = len(hc.m), hc.m.values().count(True)
  onSPg = float(nOnS)/nHc < 0.5 if nHc else True
  pg['onSubsPage'] = onSPg

  # pick fly images
  cands = (preds[onSPg] if preds else fv.inSapKeys(onSPg)) - set(hc.m.keys())
  nfis = N_ROWS * N_COLS
  if len(cands) < nfis:
    print "error: not enough fly images"
    return max(pgid-1, 1), False
  pg['flies'], pg['frames'] = nfis*[0], nfis*[0]

  for i, key in enumerate(random.sample(cands, nfis)):
    hc.m[key] = onSPg
    pg['flies'][i], pg['frames'][i] = key

  hc.d[pgid] = pg
  return pgid, True

# - - -

# analyze clusters in classifications
def analyze(fv):
  print "classification clusters:"
  cls = goodClassifications()[0]
  keys = sorted(cls.keys())
  getImg = getGetImg(fv)
  si, clusters, xyp = 0, [], None
  for i, key in enumerate(keys):
    xyc = fv.xy(key)
    if i > 0:
      last = i == len(keys)-1
      if distance(xyc, xyp) > MAX_DIST or last:
        clusters.append(keys[si:i+last])
        si = i
    xyp = xyc
  ls, tl, nSh, k2p = collections.defaultdict(int), 0, 0, key2page()
  img00 = getImg((0, 0))
  def header(txt, key):
    return "%d:r%dc%d" %k2p[key] if opts.showPageNums else txt
  scTh = 2 if opts.showSuspClusters else opts.showClusterTh
  for c in clusters:
    l = len(c)
    if l >= scTh:
      img1 = getImg(c[0])
      imgs, nm = [(img1, header('orig', c[0]))], 0
      for key in c[1:]:
        img2 = getImg(key)
        nc = normCorr(img1, img2)
        if nc < CORR_TH:
          img1 = img2.copy()
          cv2.rectangle(img2, (0, 0), bottomRight(img2), COL_R_D, 1)
          nm += 1
        imgs.append((img2, header('%.3f'%nc, key)))
      for i, (img, hdr) in enumerate(imgs):
        if cls[c[i]]:
          putText(img, 'on', (2, 2), (0, 1), TXT_STL_ON_CL)
      if not (opts.showSuspClusters and nm == l-1 and
          all(cls[k] == cls[c[0]] for k in c[1:])):
        print "%s - %s: %d, dist %.1f" %(c[0], c[-1], l,
          distance(fv.xy(c[0]), fv.xy(c[-1])))
        cv2.imshow("%s - %s" %(c[0], c[-1]),
          combineImgs(imgs, style=TXT_STL_HDR_CL)[0])
        nSh += 1
        cv2.waitKey(0)
    ls[l] += 1
    tl += l
  assert tl == len(keys)
  if nSh == 0:
    print "  none to show"
  print "\ncluster length distribution:\n  %s" \
    %", ".join("%d:%d" %(k, v) for k, v in sorted(ls.items()))

# - - -

# note: fails (in copytree) if opts.ccDataDir already exists (accidental
#  overwrite protection)
def copyCcDataDir():
  shutil.copytree(opts.ccDataDirSrc, opts.ccDataDir)
  backup(os.path.join(opts.ccDataDir, CREATION_LOG))

def ccDataFile(n):
  fn = os.path.join(opts.ccDataDir, n)
  if os.path.isfile(fn) and opts.ccDataDir != PRED_DATA_DIR and \
      not opts.ccDataDirSrc:
    raise InternalError("file %s already exists" %n)
  return fn

# predict using cuda-convnet and write predictions file
def predict(videofn, nb, keys):
  predDir = PRED_DATA_DIR+"pred/"
  print "\npredicting..."
  cmd = [SHOWNET, "-f", MODEL_DIR,
    "--write-features=probs",
    "--feature-path="+predDir,
    "--data-path="+PRED_DATA_DIR,
    "--test-range=1-%d"%nb]
  with open(LOG_FILE, 'w', 1) as lf:
    execute(cmd, CC_DIR, lf)

  allPreds, bSz = np.zeros((len(keys),), np.int), len(keys)/nb
  for i in range(nb):
    preds = readPredictionBatch(predDir, i+1)[1].argmax(axis=1)
    assert len(preds) == bSz
    allPreds[i*bSz:(i+1)*bSz] = preds

  pred2keys = invert(zip(keys, allPreds), toSet=True)
  assert sorted(pred2keys.iterkeys()) == [0, 1]
  ml = max(len(ln) for ln in LABEL_NAMES)
  for i, lbl in enumerate(LABEL_NAMES):
    print ("  %-"+str(ml+1)+"s %d") %(lbl+':', len(pred2keys[i]))

  print "writing predictions file..."
  pickle({'pred2keys': pred2keys}, predsFn(videofn), backup=True, verbose='B')

def taggedOnSDiff(pg, users):
  tagged, onSubs = [np.array([pg[u][k] for u in users])
    for k in ['tagged', 'onSubs']]
  tagged = tagged.any(axis=0)
  onSDiff = (onSubs - onSubs[0,:]).any(axis=0)
  return tagged, onSDiff

def users(pg, underscore=True): 
  si = 0 if underscore else 1
  return [k[si:] for k in pg.keys() if k.startswith('_')]

# note: nUsers: 0: all pages, None: only pages with max. number of users
def goodPages(nUsers=None):
  for pgid, pg in sorted(hc.d.iteritems()):
    usrs = users(pg)
    if nUsers is None:
      nUsers = len(usrs)
    elif nUsers > 0 and len(usrs) < nUsers:
      break
    yield pgid, pg, usrs

# returns all classifications with "on substrate" agreement and various
#  counters (see below)
def goodClassifications(excludeTagged=True, requireInPreds=True,
    allPages=False):
  cls, predsSet = {}, set.union(*preds.values()) if preds else None
  npg = nall = nd = nt = ngnp = ngdp = 0
    # numbers of: pages, all classifications, classification differences,
    #  tagged images, good but not in predictions, good but different from
    #  page-level
  for pgid, pg, users in goodPages(0 if allPages else None):
    tagged, onSDiff = taggedOnSDiff(pg, users)
    onSPg = pg['onSubsPage']
    npg += 1
    for f, fi, onS, tg, onSD in zip(
        pg['flies'], pg['frames'], pg[users[0]]['onSubs'], tagged, onSDiff):
      nall += 1
      nd += onSD and not tg
      nt += tg
      if not (tg and excludeTagged) and not onSD:   # good
        key = (f, fi)
        ngdp += int(onS != onSPg)
        if requireInPreds and predsSet and key not in predsSet:
          ngnp += 1
        else:
          cls[key] = onS
  return cls, dict(pages=npg, all=nall, diff=nd, tagged=nt,
    goodNotInPreds=ngnp, goodDiffFromPage=ngdp, users=len(users))
 
ccd = anonObj(dict(bNumOffset=0))   # shared between writeCcData() calls

# write cuda-convnet data
def writeCcData(fv, vn, last):
  fisH, bSz, nb = opts.fis/2, opts.batchSize, opts.numBatches

  if opts.predict:
    nb = nb or 6
    if nb > len(PRED_DATA_BATCH.batchNums()):
      raise InternalError("too many batches")
    print "batch size: %d, number batches: %d, total images: %d" \
      %(bSz, nb, nb*bSz)
    keys = chooseClassifications(fv,
      goodClassifications(excludeTagged=False, allPages=True)[0])
    keys = chooseImages(fv, nb*bSz, keys=keys)[0]
    cls = dict.fromkeys(keys, False)
    print
  else:
    cls, c = goodClassifications()
    if opts.ccDataDirSrc:
      srcVnKey = unpickle(ccDataFile(bmFn()))['YL_vn_key']
      ncsO = len(cls)
      cls = dict((k, v) for k, v in cls.iteritems() if (vn, k) not in srcVnKey)
    ncs, ngdp = len(cls), c['goodDiffFromPage']
    nb = nb or ncs/bSz
    print "batch size: %d, number batches: %d" %(bSz, nb)
    print "  human classifications: %s" %(ncs if not opts.ccDataDirSrc else
      "original: %d, using: %d (not in source)" %(ncsO, ncs))
    print "  error of page classifications: {:.2%}".format(float(ngdp)/ncs)

  frms = fv.getFrames([fi for f, fi in cls], channel=1)

  fimgs, minAll, maxAll, bxs = {}, 255, 0, {}
  for key in cls:
    fimg, bxs[key] = fv.getFlyImage(key, fisH, frms=frms)
    mi, ma = np.amin(fimg), np.amax(fimg)
    minAll, maxAll = min(minAll, mi), max(maxAll, ma)
    fimgs[key] = [fimg.reshape(opts.fis**2), mi, ma]
  print "fly image stats"
  print "  fly image pixel values: min=%d, max=%d" %(minAll, maxAll)
  minD = min(minAll, 255-maxAll)
  print "    (minimum distance to white or black: %d)" %(minD)
  nrmMin, nrmMax = minAll-minD, maxAll+minD
  print "  maximum extension into border: %.1f" %max(bxs.values())
  nbxWarn = np.count_nonzero(np.array(bxs.values()) >= WARN_BORDER_X)
  print "    number of images with border extension >= %d: %d" \
    %(WARN_BORDER_X, nbxWarn)
  if nbxWarn:
    print "    note: to check the fly images, replace --info or -w with --intrs"

  if opts.ccDataInfo:
    return

  print "writing cuda-convnet data..."
  assert not opts.ccDataInfo
  batch = np.zeros((opts.fis**2, bSz), np.uint8)
  labels, minMaxB, minMax, vnKey = bSz * [None], bSz * [None], [], []
  k2p = key2page()
  allKeys = [k for k in cls.keys() if (vn, k2p[k]) not in hc.errs]
  hcErrorsMsg("filtered", len(cls) - len(allKeys))
  assert len(allKeys) >= bSz*nb
  testKeys = allKeys[-bSz:]
    # note: special treatment of test keys no longer required; kept to
    #  minimize cc data changes
  bNum, i = 1, 0
  batchMeans = []
  while bNum <= nb:
    keys = allKeys[(bNum-1)*bSz:bNum*bSz] if bNum < nb else testKeys
    for key in keys:
      assert (bNum == nb) == (key in testKeys)
      fimg, mi, ma = fimgs[key]
      batch[:,i], labels[i], minMaxB[i] = fimg, int(cls[key]), (mi, ma)
      vnKey.append((vn, key))
      i += 1
      if i == bSz:   # done with current batch
        pickle({'data': batch, 'labels': labels},
          ccDataFile(dbFn(bNum + ccd.bNumOffset)))
        if bNum < nb or opts.videos:   # if video list: only training batches
          batchMeans.append(np.mean(batch, axis=1))
        minMax.append(minMaxB[:])
        i, bNum = 0, bNum+1
        break

  if opts.predict:
    assert list(zip(*vnKey)[1]) == allKeys

  # note: the following could be simplified by having the above loop append
  #  directly to ccd.batchMeans, etc.
  ccd.bNumOffset += nb
  if not hasattr(ccd, 'batchMeans'):
    ccd.batchMeans = []
    ccd.minAll, ccd.maxAll, ccd.minMax = 255, 0, []
    ccd.prc, ccd.vnKey = [], []
  ccd.batchMeans.extend(batchMeans)
  ccd.minAll, ccd.maxAll = min(ccd.minAll, minAll), max(ccd.maxAll, maxAll)
  ccd.minMax.extend(minMax)
  ccd.prc.extend([k2p[k] for vn, k in vnKey])
  ccd.vnKey.extend(vnKey)

  if last:
    if opts.predict:
      predict(fv.filename(), nb, allKeys)
    else:
      fn = ccDataFile(bmFn())
      minMaxAll = {'YL_min_max_all': (ccd.minAll, ccd.maxAll),
        'YL_min_distance_black_white': min(ccd.minAll, 255-ccd.maxAll)}
      if opts.ccDataDirSrc:
        print "\nupdating %s..." %bmFn()
        bm = unpickle(fn)
        if bm['num_cases_per_batch'] != bSz or \
            bm['label_names'] != LABEL_NAMES or bm['num_vis'] != opts.fis**2:
          raise InternalError("source directory has different data layout")
        nb, (minAll, maxAll) = len(ccd.minMax), bm['YL_min_max_all']
        print "  batches replaced: 1%s" %("-%d" %nb if nb > 1 else "")
        print "  min_all: %d -> %d" %(minAll, ccd.minAll)
        print "  max_all: %d -> %d" %(maxAll, ccd.maxAll)
        bm.update(minMaxAll)
        bm['YL_min_max'][:nb] = ccd.minMax
        bm['YL_page_row_col'][:nb*bSz] = ccd.prc
        bm['YL_vn_key'][:nb*bSz] = ccd.vnKey
        pickle(bm, fn)
      else:
        dm = np.mean(ccd.batchMeans, axis=0).astype(np.float32). \
          reshape(opts.fis**2,1)
        bm = {'num_cases_per_batch': bSz, 'label_names': LABEL_NAMES,
          'num_vis': opts.fis**2, 'data_mean': dm,
          'YL_min_max': ccd.minMax,
          'YL_page_row_col': ccd.prc, 'YL_vn_key': ccd.vnKey}
        bm.update(minMaxAll)
        pickle(bm, fn)

# - - -

def printVideoMsg(vn, all=True, first=True):
  bn = os.path.basename(FLY_VIDEOS[vn])
  if all:
    print ('' if first else '\n') + "--- video %d (%s) ---" %(vn, bn)
  else:
    print "video %d  (%s)" %(vn, bn)

def processVideos():
  if opts.writeCcData and opts.ccDataDirSrc:
    copyCcDataDir()
  lfn = ccDataFile(CREATION_LOG) if opts.writeCcData else os.devnull
  with open(lfn, 'w', 1) as lf:
    # note: move the three following lines into util.py as, e.g., logCommand()
    rev, mod = svnRevision()
    lf.write('%s revision: %s\n%s\n\n' %(mod, rev, ' '.join(sys.argv)))
    sys.stdout = Tee([sys.stdout, lf])

    vns = range(len(FLY_VIDEOS)) if opts.videos == ALL_VIDEOS else \
      (map(int, opts.videos.split(',')) if opts.videos else [opts.videoNum])
    for i, vn in enumerate(vns):
      processVideo(vn, i == 0, i == len(vns)-1)

def processVideo(vn, first, last):
  fn = FLY_VIDEOS[vn]
  printVideoMsg(vn, opts.videos, first)

  fv = FlyVideo(vn)
  if opts.playVideo:
    playVideo(fv)
    return

  fv.loadSubsMatFile(bMax=opts.bMax)
  loadPreds(fn)
  if opts.checkPreds:
    checkPreds(fv)
    return
  if opts.evalImageChoice is not None:
    evalImageChoice(fv)
    return

  ccData = opts.writeCcData or opts.ccDataInfo or opts.predict
  hcfEx = loadHumanClassification(vn,
    svnHc=(ccData or opts.analyze) and not opts.sharedHc)
  if ccData and not opts.predict and not hcfEx:
    return
  hcStats(allPages=not (opts.writeCcData or opts.ccDataInfo or opts.analyze))
  print
  if ccData:
    writeCcData(fv, vn, last)
  elif opts.analyze:
    analyze(fv)
  else:
    locker = hcLock(acquire=True)
    if locker is not None:
      print "video is locked by %s" %locker
      return
    humanClassification(fv)

# - - -

def videoList():
  print "videos:"
  for i, fn in enumerate(FLY_VIDEOS):
    print "  %d: %s" %(i, os.path.basename(fn))

def hcStatsAllVideos():
  for i, fn in enumerate(FLY_VIDEOS):
    printVideoMsg(i, first=i==0)
    loadPreds(fn)
    if loadHumanClassification(i, svnHc=not opts.sharedHc):
      hcStats()

def hcDiff():
  for i, fn in enumerate(FLY_VIDEOS):
    printVideoMsg(i, first=i==0)
    if not loadHumanClassification(i, svnHc=False):
      continue
    shr = copy.deepcopy(hc.d)
    loadHumanClassification(i, svnHc=True)
    svn, diff = hc.d, False

    print "comparing human classifications:"
    dnp = len(shr) - len(svn)
    if dnp:
      print '"shared" has %s more page%s' %(dnp, "" if dnp == 1 else "s")
      diff = True
    for pgid, pg in sorted(svn.iteritems()):
      pgSh = shr[pgid]
      if pg != pgSh:
        diff, ds, usrs, usrsSh = True, [], users(pg), users(pgSh)
        for user in usrs:
          showUser = True
          for k in ['onSubs', 'tagged']:
            for i, val in enumerate(pg[user][k]):
              if val != pgSh[user][k][i]:
                r, c = divmod(i, N_COLS)
                ds.append('%s%s r%dc%d' %(
                  '%s: ' %user[1:] if showUser else '', k, r, c))
                showUser = False
        assert len(usrs) <= len(usrsSh)
        newusrs = set(usrsSh).difference(usrs)
        if newusrs:
          ds.append("new user%s (%s)" %("" if len(newusrs) == 1 else "s",
            ", ".join(u[1:] for u in sorted(newusrs))))
        print "difference%s for page %d%s" %(
          "s" if len(ds) > 1 else ("" if ds else "(s)"), pgid,
          (": " if ds else "") + ", ".join(ds))
    if not diff:
      print "identical"

def copyPreds():
  print "copying (and gzip'ing) prediction files:"
  for i, fn in enumerate(FLY_VIDEOS):
    pfn = predsFn(fn)
    bn, exists = os.path.basename(pfn), os.path.isfile(pfn)
    print "  %d: %s%s" %(i, bn, "" if exists else "  [missing]")
    if exists:
      nfn = PREDS_SUBDIR+bn
      shutil.copy(pfn, nfn)
      gzip(nfn)

def classifyingUsers():
  d, pids = unpickle(SHARED_DATA_LOCK) or {}, set(psutil.get_pid_list())
  print "classifying users (by video):\n  %s" %(
    "none" if not d else
    "\n  ".join("%d: %s (pid %d%s)"
      %(k, u, pid, "" if pid in pids else " terminated")
      for k, (u, pid) in d.iteritems()))
 
# - - -

opts = options()
overrideOptions()
if opts.videoList:
  videoList()
elif opts.hcStats:
  hcStatsAllVideos()
elif opts.hcDiff:
  hcDiff()
elif opts.copyPreds:
  copyPreds()
elif opts.classifyingUsers:
  classifyingUsers()
else:
  processVideos()


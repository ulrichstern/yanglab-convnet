#
# common for "on substrate" classification
#
# 1 Feb 2014 by Ulrich Stern
#
# notes:
# * "key" used in FlyVideo and DataBatch is (fly, frame index)
# * inSap stands for "in substrate area polygon"

import math, os
import scipy.io
import pylru

from util import *

# - - - UI

TXT_STL_ON = (cv2.FONT_HERSHEY_PLAIN, 1, COL_G, 1, cv2.CV_AA)
TXT_STL_OFF = (cv2.FONT_HERSHEY_PLAIN, 1, COL_R, 1, cv2.CV_AA)
TXT_STL_TAG = (cv2.FONT_HERSHEY_PLAIN, 1.2, COL_Y, 1, cv2.CV_AA)

# - - - directories and similar

HOME_DIR = "/home/uli/"
EXP_DIR, EXP_LOG = HOME_DIR+"exp/cc/", "__experiment.log"
DATA_DIR = HOME_DIR+"data/onsubs/"
CONVNET, SHOWNET = "convnet.py", "shownet.py"
CC_CMDS = frozenset((CONVNET, SHOWNET))

def ccDir(dropOut=False):
  return HOME_DIR+"code/" + \
    ("dropout/cuda-convnet/trunk/" if dropOut else "cuda-convnet/")

SVN_DATA_DIR, HC_ERROR_FILE = "human-classification/", "hcErrors.csv"

def dbFn(bNum, dir=''): return os.path.join(dir, "data_batch_%d" %bNum)
def bmFn(dir=''): return os.path.join(dir, "batches.meta")

# - - -

OFF_LABEL, ON_LABEL = 'off', 'on subs'
LABEL_NAMES = (OFF_LABEL, ON_LABEL)
LABEL_SET = frozenset(LABEL_NAMES)

CONVNET_PREFIX = 'ConvNet'

# - - - exceptions

class VideoError(Error): pass

# - - - options

def addGpuOption(p):
  gpu = 0
  p.add_argument('-g', dest='gpu', type=int, metavar='ID',
    nargs='?', const=gpu, default=None,
    help='use the given GPU (device id; for -g w/out ID: %(const)s)')

# possibly adds --gpu to the given command
def addGpu(cmd, opts):
  assert cmd[0] in CC_CMDS
  if opts.gpu is not None:
    assert all(not e.startswith('--gpu') for e in cmd[1:])
    cmd += ['--gpu=%d'%opts.gpu]
  return cmd

# - - -

# returns VideoCapture for the given filename
def videoCapture(fn):
  cap = cv2.VideoCapture(fn)
  if not cap.isOpened():
    raise VideoError(
      "cannot initialize video capture for %s" %(fn) if os.path.exists(fn)
        else "%s does not exist" %(fn))
  return cap

# returns the given frame
# note: appears to work for H.264 on NZXT-U; initially assumed may need to
#  start reading some number of frames earlier
def getFrame(cap, n, channel=None, cache=None):
  if cache is not None and n in cache:
    return cache[n]
  img = readFrame(cap, n)
  if channel is not None:
    img = img[:,:,channel]
  if cache is not None:
    cache[n] = img
  return img

# returns human classification errors as set with tuples (video, (page, row,
#  column)) (and returns the filename)
def readHcErrors():
  fn = SVN_DATA_DIR + HC_ERROR_FILE
  hces = readCsv(fn, nCols=4)
  return set((vn, (pg, r, c)) for vn, pg, r, c in hces) if hces else set(), fn

# labels image "on" or "off"
def labelImg(img, onSubs, txtStl=None):
  cv2.rectangle(img, (0,0), bottomRight(img), COL_G if onSubs else COL_R, 1)
  putText(img, 'on' if onSubs else 'off', (2,2), (0,1),
    txtStl or (TXT_STL_ON if onSubs else TXT_STL_OFF))

# - - - Ctrax'ed fly video
# note: "original" video refers to version prior to conversion for tracking
#  (original video typically has color and higher frame rate and resolution)

class FlyVideo:

  _VID_DIR = "/media/YLD5/HDD-tracking/"
  _VID_DIR1 = _VID_DIR + "Uli/onsubs classification/"
  _YLD3S, _YLD3H = "/media/YLD3/SSD-tracking/", "/media/YLD3/HDD-tracking/"
  _YLD4S, _YLD4H = "/media/YLD4/SSD-tracking/", "/media/YLD4/HDD-tracking/"
  _SYN = "/media/Synology/"
  # filenames
  FNS = (
    _VID_DIR + "analysis tests/test 1/c2__2013-08-07__17-30-17 AD.avi",
    _VID_DIR1 + "2013-07-09 purple/c2__2013-07-09__16-18-37 AD.avi",
    _VID_DIR1 + "2013-05-17 white/c2__2013-05-17__17-11-15 AD.avi",
    _VID_DIR1 + "2013-05-21 purple/c1__2013-05-21__17-07-07 AD.avi",
    _VID_DIR1 + "2013-05-20 purple/c4__2013-05-20__16-40-04 AD.avi",
    _VID_DIR1 + "2013-05-17 white/c4__2013-05-17__17-11-14 AD.avi",   # 5
    _VID_DIR1 + "2013-05-17 white/c3__2013-05-17__17-11-14 AD.avi",
    _VID_DIR1 + "2013-05-20 white/c3__2013-05-20__16-40-46 AD.avi",
    _VID_DIR1 + "2013-07-08 purple/c2__2013-07-08__16-48-33 AD.avi",
    _VID_DIR1 + "2013-07-09 purple/c1__2013-07-09__16-18-37 AD.avi",
    _VID_DIR1 + "2013-07-10 purple/c1__2013-07-10__15-32-12 AD.avi",   # 10
    _VID_DIR1 + "2013-10-22 white night/c2__2013-10-22__17-41-03 AD.avi",
    _VID_DIR1 + "2013-10-22 white night/c3__2013-10-22__17-41-04 AD.avi",
    _VID_DIR1 + "2013-10-22 white night/c4__2013-10-22__17-41-04 AD.avi",
    _VID_DIR1 + "MB-ablated/c2__2014-05-22__16-54-06 AD.avi",
    _VID_DIR1 + "MB-ablated/c3__2014-05-24__10-24-06 AD.avi",   # 15
    _VID_DIR1 + "WT/c3__2014-05-20__16-24-19 AD.avi",
    _VID_DIR1 + "WT/c4__2014-05-20__16-24-19 AD.avi",
    _VID_DIR1 + "WT/c3__2014-05-21__16-38-36 AD.avi",
    _VID_DIR1 + "WT/c4__2014-05-21__16-38-36 AD.avi",
  )

  EGG_FILE = "egg.txt"

  _FRAMES_SKIPPED = 3   # note: better to determine from videos?

  def __init__(self, vn=None, fn=None, orig=False):
    fn = fn or self.FNS[vn]
    if vn is None and fn in self.FNS:
      vn = self.FNS.index(fn)
    self.vn, self.fn, self.orig = vn, fn, orig
    self.cap, self.fps = self._capFps(fn)
    self.imgHW = getFrame(self.cap, 0).shape[:2]
    fnO = re.sub(r' AD\.avi$', '.avi', fn)   # original video
    if os.path.isfile(fnO):
      self.capO, self.fpsO = self._capFps(fnO)
      self._setFctrs()
    else:
      self.capO = self.fpsO = self.fpsFctr = self.sizeFctr = None
    self.x, self.y, self.a, self.b, self.theta = \
      self.loadCtraxMatFile(fn, self.cap)
    self.fiCache = pylru.lrucache(100)

  def _capFps(self, fn):
    cap = videoCapture(fn)
    return cap, frameRate(cap)

  def _sizeFctr(self, img):
    fs = np.array(img.shape[:2], float) / self.imgHW
    if fs[0] != fs[1]:
      raise VideoError("size factors differ: %.2f and %.2f" %tuple(fs))
    return fs[0]

  def _setFctrs(self):
    self.fpsFctr = self.fpsO/self.fps
    self.sizeFctr = self._sizeFctr(getFrame(self.capO, 0))
 
  _fvs = {}

  # returns the instance for the given video number
  @staticmethod
  def getInstance(vn):
    fv = FlyVideo._fvs.get(vn)
    if fv is None:
      fv = FlyVideo(vn)
      FlyVideo._fvs[vn] = fv
    return fv

  # sets whether to use original video
  def useVideo(self, orig): self.orig = orig

  # returns x and y given key
  def xy(self, key): return self.x[key], self.y[key]
  # returns the video number
  def videoNum(self): return self.vn
  # returns the video filename
  def filename(self, newExt=None):
    return self.fn if newExt is None else self.changeExt(self.fn, newExt)
  # returns the length of the trajectory
  def tlen(self): return self.x.shape[1]
  # returns the frame rate factor (original/Ctrax)
  def frameRateFactor(self): return self.fpsFctr
  # returns the (frame) size factor (original/Ctrax)
  def sizeFactor(self): return self.sizeFctr

  # returns the VideoCapture object
  def capture(self): return self.capO if self.orig else self.cap
  # returns the frame rate in fps
  def frameRate(self): return self.fpsO if self.orig else self.fps
  # returns the frame count
  # TODO: make sure last frame can be read (currently no true for, e.g., v27)
  #  "so-so solution": subtract 4
  def frameCount(self): return frameCount(self.capture())

  # returns the time for the given frame in HH:MM:SS format
  def fi2time(self, fi): return frame2time(fi, self.frameRate())
  # converts the given frame index
  def fiConv(self, fi, toOrig=False, check=True):
    if toOrig:
      return intR(fi*self.fpsFctr)+self._FRAMES_SKIPPED
    else:
      fiC = intR((fi-self._FRAMES_SKIPPED)/self.fpsFctr)
      if check and fiC < 0:
        raise ArgumentError("conversion of %d yields %d" %(fi, fiC))
      return fiC
  def _fiCtrax(self, fi, check=True):
    return self.fiConv(fi, check=check) if self.orig else fi

  # reads from the underlying VideoCapture
  def read(self): return self.capture().read()
  # returns the given frame
  def getFrame(self, fi): return getFrame(self.capture(), fi)

  def _xcYcBx(self, f, fi, img, fisH, doAdjust=False):
    xc, yc = self.xy((f, fi))
    xcA = min(max(xc, fisH), img.shape[1]-fisH)
    ycA = min(max(yc, fisH), img.shape[0]-fisH)
    bx = max(abs(xc-xcA), abs(yc-ycA))
    return (xcA, ycA) if doAdjust else (xc, yc), bx

  # returns fly image and how much fly image extends into frame border
  def getFlyImage(self, key, fisH, frms=None, channel=None):
    ckey = (key, fisH)
    if ckey in self.fiCache:
      return self.fiCache[ckey]
    else:
      f, fi = key
      img = getFrame(self.cap, fi, channel) if frms is None else frms[fi]
      (xc, yc), bx = self._xcYcBx(f, fi, img, fisH)
      bxi = int(math.ceil(bx))
      if bxi > 0:
        img = cv2.copyMakeBorder(img, bxi, bxi, bxi, bxi, cv2.BORDER_REPLICATE)
        xc, yc = xc + bxi, yc + bxi
      val = (img[yc-fisH:yc+fisH, xc-fisH:xc+fisH], bx)
      self.fiCache[ckey] = val
      return val

  def _scaleFctr(self): return self.sizeFctr if self.orig else 1

  # draws ellipse for the given fly and frame index
  # note: cols gives colors for bottom and top
  def ellipse(self, img, f, fi, cols=None, txt=None, style=None):
    col = COL_Y if cols is None else cols[self.inTop(f, fi)]
    fi = self._fiCtrax(fi, check=False)
    if 0 <= fi < self.x.shape[1]:
      s = self._sizeFctr(img)
      pos = intR(self.x[f,fi]*s, self.y[f,fi]*s)
      axs = intR(self.a[f,fi]*s, self.b[f,fi]*s)
      cv2.ellipse(img, pos, axs, self.theta[f,fi], 0, 360, col, 1)
      if txt is not None:
        if style is None:
          raise ArgumentError("text given but no style")
        putText(img, txt, tupleAdd(pos, (0, axs[0]+3)), (-.5, 1), style)

  def _setXYMR(self):
    if not hasattr(self, 'yRng'):
      ymi, yma = self.y.min(), self.y.max()
      assert yma - ymi > 100
      self.yMid, self.yRng = (ymi+yma)/2, (ymi, yma)
      f1 = self.flyIdx(1)
      x1ma, x2mi = self.x[f1].max(), self.x[1-f1].min()
      assert x1ma+.1 < x2mi
      self.xMid = (x1ma+x2mi)/2

  # returns the y range (ymin, ymax)
  def yRange(self):
    self._setXYMR()
    return self.yRng
  # returns an x value that can separate the flies
  def xSeparator(self):
    self._setXYMR()
    return self.xMid

  # returns whether fly is in "top" part of the chamber in the given frame and
  #  boolean ndarray for all flies and frames if no fly is given
  # note: uses average of ymin and ymax
  def inTop(self, f=None, fi=None):
    self._setXYMR()
    return self.y < self.yMid if f is None else \
      self.y[f, self._fiCtrax(fi)] < self.yMid

  def _setFnFMaps(self):
    if not hasattr(self, 'fn2f'):
      idxs = [f for x, f in sorted(zip(self.x[:,0], range(self.x.shape[0])))]
      self.fn2f = dict((i+1, idx) for i, idx in enumerate(idxs))
      self.f2fn = invert(self.fn2f)

  # returns the fly index given the "human" fly number (1: left, 2: right)
  def flyIdx(self, flyNum):
    self._setFnFMaps()
    return self.fn2f[flyNum]
  # inverse of flyIdx()
  def flyNum(self, flyIdx=None, x=None):
    if flyIdx is not None:
      self._setFnFMaps()
      return self.f2fn[flyIdx]
    else:
      if x is None:
        raise ArgumentError("flyIdx or x must be given")
      self._setXYMR()
      return 1 if x < self.xMid*self._scaleFctr() else 2

  # changes extension of the given video filename (e.g., to '.mat')
  @staticmethod
  def changeExt(videofn, toExt):
    return replaceCheck(AVI_X, toExt, videofn)

  # returns x, y, a, b, theta (in degrees) from Ctrax MAT-file
  @staticmethod
  def loadCtraxMatFile(videofn, cap, verbose=False):
    fn = fn1 = FlyVideo.changeExt(videofn, '.mat')
    while True:
      ffn = addPrefix(fn, "fixed_")
      if os.path.isfile(ffn):
        fn = ffn
      else:
        break
    if fn != fn1:
      print "note: using fixed MAT-file (%s)" %basename(fn)
    if verbose:
      print "loading %s" %fn
    m = scipy.io.loadmat(fn)
    if 'ntargets' in m:
      # regular MAT-file:
      # m['x_pos'] is, e.g., (431992, 1) ndarray with x-pos of all flies in
      #   order frame 1 fly 1, frame 1 fly 2, ..., frame 2 fly 1, ...
      # m['ntargets'] is, e.g., (215996, 1) ndarray with the numbers of flies
      #   for each frame
      if np.count_nonzero(m['ntargets'] != 2):
        raise InternalError('each frame must have 2 flies')
      x, y, a, b, theta = [np.transpose(m[n].reshape((-1,2))) for n in
        ['x_pos', 'y_pos', 'maj_ax', 'min_ax', 'angle']]
    else:
      # fixed MAT-file:
      # m['trx'] is, e.g., (1, 2) ndarray in case of 2 flies
      # m['trx'][0,0] is numpy.void of length 21
      # m['trx'][0,0]['x'] is, e.g., (1, 215996) ndarray
      t = m['trx']
      if t.size != 2:
        raise InternalError('number of flies must be 2')
      x = np.zeros((t.size, t[0,0]['x'].size))
      y, a, b, theta = (x.copy() for i in range(4))
      for f in range(t.size):
        x[f,:], y[f,:] = t[0,f]['x'][0,:], t[0,f]['y'][0,:]
        a[f,:], b[f,:] = t[0,f]['a'][0,:], t[0,f]['b'][0,:]
        theta[f,:] = t[0,f]['theta'][0,:]
    a, b = [v*2 for v in [a, b]]
    theta = theta * 180/np.pi

    # flip y coordinate (Ctrax bug)
    if fourcc(cap) == 'MJPG':
      h = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
      y = h - y
      theta = -theta

    if verbose:
      print "  max(a)=%.1f, max(b)=%.1f" %(np.max(a), np.max(b))

    return x, y, a, b, theta

  # sets (and returns) inSap and onSubs from "on substrate" MAT-file
  #  bMax: non-None: print value from file, > 0: override value from file
  def loadSubsMatFile(self, videofn=None, bMax=None, verbose=False):
    if hasattr(self, 'inSap'):
      return
    fn = self.changeExt(videofn or self.fn, ' subs.mat')
    if verbose:
      print "loading %s" %fn
    m = scipy.io.loadmat(fn)
    # m['inSap'] is, e.g., (1, 2) ndarray in case of 2 flies
    #   m['inSap'][0,0] is, e.g., (1, 215996) uint8 ndarray that gives
    #     "in substrate area polygon" for each frame
    # m['onSubs'][0,0] is, e.g., (1, 215996) uint8 ndarray that gives
    #   "on substrate" for each frame
    # m['regs'][0,0] is, e.g., (1, 215996) uint8 ndarray that gives
    #   region (1:top, 2:middle, 3:bottom) for each frame
    # m['bMax'][0,0] is, e.g., 2.1  (same for 'abMin')
    inSap = np.zeros((m['inSap'].size, m['inSap'][0,0].size), np.bool)
    onSubs = inSap.copy()
    regs = np.zeros_like(inSap, np.uint8) if 'regs' in m else None
    for f in range(inSap.shape[0]):
      inSap[f,:] = m['inSap'][0,f][0,:]
      onSubs[f,:] = m['onSubs'][0,f][0,:]
      if regs is not None:
        regs[f,:] = m['regs'][0,f][0,:]
    if regs is not None:
      inTop = self.inTop()
      assert np.all(inTop[regs == 1]) and np.all(inTop[regs == 3] == False)

    if 'bMax' in m and bMax is not None:
      bMx, abMin = [m[n][0,0] for n in ['bMax', 'abMin']]
      print '"on substrate" MAT-file:\n bMax=%.2f, abMin=%.2f\n' %(bMx, abMin)

    # adjust bMax
    if bMax > 0:
      if 'abMin' not in locals(): abMin = 2.1
      newOnSubs = np.logical_and(b < bMax*2, a/b > abMin)
      newOnSubs = np.logical_and(inSap, newOnSubs)
      print 'using bMax=%.2f -- number of "on substrate" changes: %d\n' \
        %(bMax, np.count_nonzero(newOnSubs != onSubs))
      onSubs = newOnSubs

    self.inSap, self.onSubs, self.regs = inSap, onSubs, regs
    return inSap, onSubs

  # returns keys that are inSap as set or, if onSub is not None, "on" or
  #  "off" (according to "on substrate" MAT-file)
  def inSapKeys(self, onSub=None):
    self.loadSubsMatFile()
    return set(zip(*(
      self.inSap if onSub is None else
        self.inSap & (self.onSubs if onSub else ~self.onSubs) ).nonzero()))

  # returns the Ctrax regions (possibly None)
  def regions(self):
    self.loadSubsMatFile()
    return self.regs

  # returns map from frame index to frame
  def getFrames(self, fis, channel=None):
    if len(fis) > 100:
      print "loading frames..."
    frms = {}
    for fi in sorted(fis):
      frms[fi] = getFrame(self.cap, fi, channel)
    return frms

  # guarantees that flies are increasing and that egg times are increasing
  #  for each fly
  def _checkEggs(self, eggs):
    fnes = [(self.flyNum(f), fi) for f, fi in eggs]
    bad = [e for e, en in zip(fnes[:-1], fnes[1:])
      if not (e[0] <= en[0] and (e[1] < en[1] or e[0] != en[0]))]
    if bad:
      error("bad egg order: %s" %", ".join(
        "f%d %s" %(fn, self.fi2time(fi)) for fn, fi in bad))

  # returns top/bottom egg numbers tuple by fly and possibly reports the numbers
  def eggStats(self, eggs, report=False):
    ft, es = [(f, self.inTop(f, fi)) for f, fi in eggs], []
    for f in (0, 1):
      t, b = (sum(t == top for fl, t in ft if fl == f) for top in (True, False))
      if t + b > 0 and report:
        print "  fly %d eggs: top: %d, bottom: %d" %(self.flyNum(f), t, b)
      es.append((t, b))
    return es

  # returns list with keys of eggs (keys use Ctrax frame indexes)
  # TODO: store eggs in FlyVideo and move more egg-related code here
  def loadEggFile(self, eggFn=None, allowNoFile=False):
    dn, ef = os.path.dirname(self.fn), eggFn or self.EGG_FILE
    if ef == self.EGG_FILE and not os.path.isfile(os.path.join(dn, ef)):
      ef = replaceCheck(TXT_X, "-v%d.txt" %self.vn, ef)
    ep = checkIsfile(dn, ef, noError=allowNoFile)
    if not ep:
      print "no egg file"
      return []
    fps = self.frameRate()
    eggs = [(self.flyIdx(int(fn)), int(round(time2s(tm)*fps)))
      for fn, tm in readCsv(ep, toInt=False, nCols=2)]
    self._checkEggs(eggs)
    print "egg file: %s" %nItems(len(eggs), "egg")
    self.eggStats(eggs, report=True)
    return eggs

# - - - cuda-convnet data
# format: see Alex Krizhevsky's CIFAR-10 dataset

# one or more data batches
class DataBatch:

  _AUTO_RESIZE = {92:64}
  _BATCH_NUM = re.compile(r'^data_batch_(\d+)$')

  # ctor; loads data batches with the given numbers
  def __init__(self, dataDir, bNums=None, labelSet=None):
    if bNums is None: bNums = self.batchNumsForDir(dataDir)
    self.batches = {}
    for i, bNum in enumerate(bNums):
      batch = self.read(dataDir, bNum)
      self.batches[bNum] = batch
      isnc = self.imgSizeNumChannels(batch[0])
      if i == 0:
        self.imgSzNCh = isnc
        nis = self._AUTO_RESIZE.get(isnc[0])
        self.newImgSz = nis if nis else isnc[0]
      else:
        assert self.imgSzNCh == isnc

    bm = unpickle(bmFn(dataDir))
    self.labelNames = tuple(bm['label_names'])
    self._batchSize = bm['num_cases_per_batch']
    self._pageRowCol, self._vnKey = (bm.get(key) for key in
      ('YL_page_row_col', 'YL_vn_key'))
    isnc = self.imgSizeNumChannels(nr=bm['num_vis'])
    assert labelSet is None or set(self.labelNames) == labelSet
    if hasattr(self, 'imgSzNCh'):
      assert self.imgSzNCh == isnc
    else:
      self.imgSzNCh = isnc
    self._batchNums = self.batchNumsForDir(dataDir)

  # returns image size and number of channels
  @staticmethod
  def imgSizeNumChannels(data=None, nr=None):
    if data is not None: nr = data.shape[0]
    sr = int(math.sqrt(nr))
    if nr == sr**2:
      return sr, 1
    else:
      assert nr % 3 == 0
      nr = nr/3
      sr = int(math.sqrt(nr))
      assert nr == sr**2
      return sr, 3

  # returns image size (in one dimension; height = width)
  def imgSize(self): return self.imgSzNCh[0]
  def newImgSize(self): return self.newImgSz

  # returns batch size (in images)
  def batchSize(self): return self._batchSize

  # returns the batch numbers (1, 2, ...) as sorted list
  def batchNums(self): return self._batchNums

  @staticmethod
  def batchNumsForDir(dataDir):
    return sorted(int(n) for n in
      multiMatch(DataBatch._BATCH_NUM, os.listdir(dataDir)))

  # reads data batch and returns data and labels
  @staticmethod
  def read(dataDir, bNum):
    d = unpickle(dbFn(bNum, dataDir))
    return [d[k] for k in ('data', 'labels')]

  # returns image (in OpenCV "format")
  @staticmethod
  def getImageFromData(data, idx, imgSz=None, nCh=None):
    if imgSz is None:
      imgSz, nCh = imgSizeNumChannels(data)
    return data[:,idx].reshape(nCh, imgSz, imgSz). \
      swapaxes(0,2).swapaxes(0,1)[:,:,::-1]

  def getImage(self, bNum, idx, autoResize=True, bgr=True,
      label=False, onSubs=None):
    img = self.getImageFromData(self.batches[bNum][0], idx, *self.imgSzNCh)
    nis, ois = self.newImgSz, self.imgSzNCh[0]
    if autoResize and nis != ois:
      d = (ois-nis)/2
      img = img[d:d+nis, d:d+nis]
    if bgr and img.shape[2] == 1:
      img = toColor(img)
    if label:
      if onSubs is None:
        onSubs = self.onSubstrate(bNum, idx)
      labelImg(img, onSubs)
    return img

  # returns label name
  def getLabel(self, bNum, idx):
    return self.labelNames[self.batches[bNum][1][idx]]
  def getLabels(self): return self.labelNames

  # returns whether the fly is on substrate
  def onSubstrate(self, bNum, idx):
    assert self.labelNames == LABEL_NAMES
    return self.batches[bNum][1][idx] == 1

  # returns (fly image) page, row, and column
  def pageRowColumn(self, bNum, idx):
    return self._pageRowCol[(bNum-1)*self._batchSize + idx] \
      if self._pageRowCol else None
  # returns video number and key
  def vnKey(self, bNum, idx):
    return self._vnKey[(bNum-1)*self._batchSize + idx] if self._vnKey else None

# - - - experiments

# returns list of experiment directories sorted by name
def allExpDirs():
  return [d for d in sorted(os.listdir(EXP_DIR)) if os.path.isdir(EXP_DIR+d)]

# returns, e.g., most recent experiment directory for "1"
def expandExpDir(ed, aeds=None):
  aeds = aeds or allExpDirs()
  return aeds[-int(ed)] if re.match(DIGITS_ONLY, ed) else ed

# returns experiment path (ending in '/')
def expPath(ed):
  return EXP_DIR+ed+'/'

# - - - cuda-convnet utils

# returns labels and predictions (or "features")
# note: dtype: float32;
#  shape (for batch size 400): labels: (400,), preds: (400, 2)
#  values: labels: 0, 1, ...
def readPredictionBatch(predDir, bNum):
  d = unpickle(dbFn(bNum, predDir))
  lbls, preds = [d[k] for k in ['labels', 'data']]
    # lbls.shape is, e.g., (1, 400)
  return (lbls[0], preds)

# average over multiple (dtMult) views of each fly image
def multiviewAverage(lbls, preds, dtMult):
  if dtMult == 1:
    return lbls, preds
  # sample shapes: lbls: (400*dtMult,), preds: (400*dtMult, 2)
  #  increasing index by 1 gives a different fly image 
  lbls, preds = lbls.reshape(dtMult, -1), preds.reshape(dtMult, -1, 2)
  assert np.all(lbls-lbls[0] == 0)
  return lbls[0], preds.sum(axis=0) / dtMult

CC_OPTS_EX = re.compile(r'(?<= )--([a-z]+(?:-[a-z]+)*)=(.*?)(?: |$)', re.M)
# returns dict with cuda-convnet options used in experiment
def ccOpts(ep, logfn=None):
  with open(os.path.join(ep, logfn if logfn else EXP_LOG)) as lf:
    d = {}
    for n, v in re.findall(CC_OPTS_EX, lf.read(4096)):
      if n not in d:
        d[n] = v
    return d

# returns data dir and DataBatch
def dataDirBatch(ep):
  dtDir = ccOpts(ep)['data-path']
  return dtDir, DataBatch(dtDir, labelSet=LABEL_SET)

# - - -

if __name__ == "__main__":
  print "TO DO: add tests"


#
# utilities
#
# 4 Aug 2013 by Ulrich Stern
#

import cv2
import numpy as np, math
import cPickle
import os, platform, sys, subprocess, csv, shutil
import re, operator, time, inspect, blist, bisect, collections, hashlib
import itertools
from PIL import Image, ImageFile

import util

# - - -

# OpenCV-style (BGR)
COL_W = 3*(255,)
COL_BK = (0,0,0)
COL_B, COL_B_L, COL_B_D = (255,0,0), (255,64,64), (192,0,0)
COL_G, COL_G_L, COL_G_D = (0,255,0), (64,255,64), (0,192,0)
COL_G_DD, COL_G_D224, COL_G_D96 = (0,128,0), (0,224,0), (0,96,0)
COL_R, COL_R_L, COL_R_D = (0,0,255), (64,64,255), (0,0,192)
COL_Y = (0,255,255)
COL_O = (0,127,255)

JPG_X = re.compile(r'\.jpg$', re.IGNORECASE)
AVI_X = re.compile(r'\.avi$', re.IGNORECASE)
TXT_X = re.compile(r'\.txt$', re.IGNORECASE)
DIGITS_ONLY = re.compile(r'^\d+$')
SINGLE_UPPERCASE = re.compile(r'([A-Z])')

MODULE_DIR = os.path.dirname(os.path.realpath(__file__))

_PFS = platform.system()
MAC, WINDOWS = _PFS == 'Darwin', _PFS == 'Windows'

DEVNULL_OUT = open(os.devnull, 'w')
  # note: for devnull, not worth to close file in atexit handler

# - - -

# skip import for certain packages on certain platforms, e.g., due to
#  difficult install (reduces functionality)
if not WINDOWS:
  from qimage2ndarray import array2qimage

# - - -

_READ_SIZE = 8192

# - - - exceptions

# base exception
class Error(Exception): pass

# for internal errors
class InternalError(Error): pass
# bad argument(s), CSV data
class ArgumentError(Error): pass
class CsvError(Error): pass
# possible problem with video
class VideoError(Error): pass
# to break out of nested loops
class NestedBreak(Error): pass

# - - - general

# checks that the given function returns the given value
def test(func, args, rval):
  rv = func(*args)
  if not np.all(rv == rval):
    print "args: %s\n  rval: %s\n  correct: %s" %(args, rv, rval)
    raise InternalError()

# prints warning
def warn(msg):
  print "warning: %s" %msg

# prints message and exits
def error(msg):
  print "error: %s" %msg
  sys.exit(1)

# prints the given string without newline and flushes stdout
def printF(s):
  print s,
  sys.stdout.flush()

# returns file path (joining arguments) if file exists; otherwise, exits
#  with error message or returns False (for noError=True)
# note: typical call: fp = checkIsfile(dir, fn)
def checkIsfile(*fp, **kwargs):
  fp = os.path.join(*fp)
  if not os.path.isfile(fp):
    if kwargs.get('noError'):
      return False
    error("%s does not exist" %fp)
  return fp

# shorthand for checkIsfile(*fp, noError=True)
def isfile(*fp): return checkIsfile(*fp, noError=True)

# returns anonymous object with the given attributes (dictionary)
# note: cannot be pickled
def anonObj(attrs=None):
  return type('', (), {} if attrs is None else attrs)

# returns rounded ints (as tuple if there is more than one) for the given
#  values; can be passed iterable with values
def intR(*val):
  if val and isinstance(val[0], collections.Iterable):
    val = val[0]
  t = tuple(int(round(v)) for v in val)
  return t if len(t) > 1 else t[0]

# returns the distance of two points
def distance(pnt1, pnt2):
  return np.linalg.norm(np.array(pnt1)-pnt2)

# returns svn revision number with -M appended if modified (or None if svn
#  fails) and file name of the calling module or of the given file
_GET_SVN_REV = re.compile(r'^Revision:\s+(\d+)$', re.M)
_GET_SVN_CS = re.compile(r'^Checksum:\s+([0-9a-f]+)$', re.M)
def svnRevision(fn=None):
  if fn is None:
    fn = inspect.getmodule(inspect.stack()[1][0]).__file__   # caller
  try:
    out = subprocess.check_output(["svn", "info", fn], stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError:
    return None, fn
  revs, css = [re.findall(r, out) for r in [_GET_SVN_REV, _GET_SVN_CS]]
  assert len(revs) == 1 and (revs[0] == '0' or len(css) == 1)
  return revs[0] + ('' if not css or css[0] == md5(fn) else '-M'), fn

# write to both console and file
# usage: sys.stdout = Tee([sys.stdout, logfile])
# name inspired by tee command
class Tee(list):
  def write(self, obj):
    for s in self:
      s.write(obj)
  def flush(self):
    for s in self:
      s.flush()

# simple timer
# note: time.clock() recommended for benchmarking Python, but it excludes
#  time taken by "external" processes on Ubuntu
class Timer():
  def __init__(self, useClock=True):
    self.getTime = time.clock if useClock else time.time
    self.start = self.getTime()
  # get elapsed time in seconds (as float)
  def get(self, restart=True):
    now = self.getTime()
    elp = now - self.start
    if restart: self.start = now
    return elp

# option overrider
class Overrider():
  def __init__(self, opts):
    self.opts = opts
    self.ovs = []
  # override the given option
  def override(self, name, val=True, descr=None):
    if not hasattr(self.opts, name):
      raise ArgumentError(name)
    descr = descr or SINGLE_UPPERCASE.sub(r' \1', name).lower()
    self.ovs.append((descr, val, getattr(self.opts, name)))
    setattr(self.opts, name, val)
  # report which options were overridden
  def report(self):
    if self.ovs:
      print "overrode options:\n  %s\n" %"\n  ".join(
        '"%s" to %s (from %s)' %e for e in self.ovs)

# executes the given command, raising subprocess.CalledProcessError in case
#  of problems
def execute(cmd, wd=None, logfile=DEVNULL_OUT, pythonDetect=True):
  if pythonDetect and cmd[0] != "python" and cmd[0].endswith(".py"):
    cmd = ["python"] + cmd
  logfile.write("[%s]  %s\n" %(time2str(), " ".join(cmd)))
  subprocess.check_call(cmd, stdout=logfile, stderr=logfile, cwd=wd)

# returns min and index of min
def minIdx(vals):
  return min(enumerate(vals), key=operator.itemgetter(1))[::-1]
# returns max and index of max
def maxIdx(vals):
  return max(enumerate(vals), key=operator.itemgetter(1))[::-1]

# - - - strings

# converts the given seconds since epoch to YYYY-MM-DD HH:MM:SS format
# notes:
# * current time is used if no seconds since epoch are given
# * use UTC, e.g., to convert seconds to HH:MM:SS (see test)
def time2str(secs=None, format='%Y-%m-%d %H:%M:%S', utc=False):
  if secs is None:
    secs = time.time()
  return time.strftime(format,
    time.gmtime(secs) if utc else time.localtime(secs))

def _time2strTest():
  ts = [([3601, '%H:%M:%S', True], '01:00:01')]
  for args, r in ts:
    test(time2str, args, r)

# converts the given frame index to HH:MM:SS
def frame2time(fi, fps): return time2str(fi/fps, '%H:%M:%S', utc=True)
 
# returns the number of seconds for the given time in HH:MM:SS format
def time2s(hms, format='%H:%M:%S'):
  try:
    s = time.strptime(hms, format)
  except ValueError:
    raise ArgumentError(hms)
  return s.tm_hour*3600 + s.tm_min*60 + s.tm_sec

def _time2sTest():
  ts = [('1:01:02', 3662), ('10:00:00', 36000)]
  for hms, s in ts:
    test(time2s, [hms], s)

# returns the numbers of seconds for the given time interval in HH:MM-HH:MM
#  format
# note: uses error() instead of raising ArgumentError
def interval2s(iv):
  try:
    s = [time2s(p, format='%H:%M') for p in iv.split('-')]
  except ArgumentError:
    error('cannot parse times in interval "%s"' %iv)
  if len(s) != 2:
    error('interval "%s" does not have two times separated by "-"' %iv)
  if s[0] >= s[1]:
    error('start time in interval "%s" not before end time' %iv)
  return s

# removes '/' from end of string if present
def removeSlash(s):
  return s[:-1] if s.endswith('/') else s

# similar to os.path.basename but returns, e.g., 'bar' for '/foo/bar/'
def basename(s):
  return os.path.basename(removeSlash(s))

# adds prefix to filename (base name)
def addPrefix(path, prefix):
  dn, bn = os.path.split(path)
  return os.path.join(dn, prefix+bn)

# returns plural "s"
def pluralS(n):
  return "" if n == 1 else "s"
# returns "n items" with proper plural
def nItems(n, item):
  return "%d %s%s" %(n, item, pluralS(n))

# joins, e.g., list of ints
def join(withStr, l):
  return withStr.join(str(e) for e in l)

# replaces the given pattern with the given replacement and checks that
#  the resulting string is different from original
def replaceCheck(pattern, repl, string):
  s = re.sub(pattern, repl, string)
  if s == string:
    raise ArgumentError(s)
  return s

def _replaceCheckTest():
  try:
    replaceCheck(AVI_X, ".bar", "foo")
  except ArgumentError:
    pass
  test(replaceCheck, [AVI_X, ".bar", "foo.avi"], "foo.bar")

# matches pattern against the given list of strings and returns list with
#  first subgroups
def multiMatch(pattern, l):
  if isinstance(pattern, str):
    pattern = re.compile(pattern)
  mos = (pattern.match(s) for s in l)
  return [mo.group(1) for mo in mos if mo]

def _multiMatchTest():
  ts = [([r'v(\d+)$', ['v1', ' v2', 'v3 ', 'v44']], ['1', '44'])]
  for args, r in ts:
    test(multiMatch, args, r)

# - - - tuples

# returns t2 as tuple
#  if t2 is int, float, or string, t2 is replicated len(t1) times
#  otherwise, t2 is passed through
def _toTuple(t1, t2):
  if isinstance(t2, (int,float,str)):
    t2 = len(t1) * (t2,)
  return t2

# applies the given operation to the given tuples; t2 can also be number
def tupleOp(op, t1, t2):
  return tuple(map(op, t1, _toTuple(t1, t2)))

# tupleOp() add
def tupleAdd(t1, t2): return tupleOp(operator.add, t1, t2)

# tupleOp() subtract
def tupleSub(t1, t2): return tupleOp(operator.sub, t1, t2)

# tupleOp() multiply
def tupleMul(t1, t2): return tupleOp(operator.mul, t1, t2)

# - - - blist

# returns the n elements before and after the given object in the given
#  sorted list or blist.sortedlist
def beforeAfter(sl, obj, n=1):
  blsl = isinstance(sl, blist.sortedlist)
  li = sl.bisect_left(obj) if blsl else bisect.bisect_left(sl, obj)
  ri = sl.bisect(obj) if blsl else bisect.bisect(sl, obj)
  res = 2*n*[None]
  for i in range(-n, n):
    i1 = li+i if i < 0 else ri+i
    res[i+n] = None if i1 < 0 or i1 >= len(sl) else sl[i1]
  return res

def _beforeAfterTest():
  l = [1, 4, 4, 6, 10]
  d = {
    4: [1, 6], 1: [None, 4], 0: [None, 1], 8: [6, 10], 20: [10, None],
    (4, 2): [None, 1, 6, 10], (6, 3): [1, 4, 4, 10, None, None] }
  for i in [0, 1]:
    if i == 1: l = blist.sortedlist(l)
    for obj, rv in d.iteritems():
      args = [l] + list(obj) if isinstance(obj, tuple) else [l, obj]
      test(beforeAfter, args, rv)

# - - - dictionary

# inverts mapping; if mapping has non-unique values, possibly use toSet
def invert(m, toSet=False):
  it = m.iteritems() if isinstance(m, dict) else iter(m)
  if toSet:
    d = collections.defaultdict(set)
    for k, v in it:
      d[v].add(k)
    return d
  else:
    return dict((v, k) for k, v in it)

def _invertTest():
  ms = [
    [{1:1, 2:2, 3:1}, True, {1:set([1,3]), 2:set([2])}],
    [[(1,4),(2,5)], False, {4:1, 5:2}] ]
  for m, toSet, invM in ms:
    test(invert, [m, toSet], invM)

# - - - file

# returns absolute path for file that is part of package given relative name
def packageFilePath(fn):
  return os.path.join(MODULE_DIR, fn)

# backs up (by renaming or copying) the given file
def backup(fn, verbose=False, copy=False):
  if os.path.isfile(fn):
    if verbose:
      print "backing up %s" %os.path.basename(fn)
    fn1 = fn+'.1'
    if copy:
      shutil.copyfile(fn, fn1)
    else:
      if os.path.isfile(fn1):
        os.remove(fn1)
      os.rename(fn, fn1)

# saves the given object in the given file, possibly creating backup
def pickle(obj, fn, backup=False, verbose=''):
  if backup:
    util.backup(fn, 'B' in verbose)
  f = open(fn, 'wb')
  cPickle.dump(obj, f, -1)
  f.close()

# loads object from the given file
def unpickle(fn):
  if not os.path.isfile(fn):
    return None
  f = open(fn, 'rb')
  obj = cPickle.load(f)
  f.close()
  return obj

# returns MD5 of the given file
def md5(fn):
  m = hashlib.md5()
  with open(fn, 'rb') as f:
    while True:
      data = f.read(_READ_SIZE)
      if not data: break
      m.update(data)
    return m.hexdigest()

# gzips the given file, overwriting a possibly existing ".gz" file
def gzip(fn):
  execute(['gzip', '-f', fn])

# reads CSV file skipping comment rows
def readCsv(fn, toInt=True, nCols=None):
  if not os.path.isfile(fn): return None
  with open(fn) as f:
    dt = list(csv.reader(row for row in
      (row.partition('#')[0].strip() for row in f) if row))
    if nCols:
      for row in dt:
        if len(row) != nCols:
          raise CsvError("nCols=%d, row=%s" %(nCols, "|".join(row)))
    return [map(int, row) for row in dt] if toInt else dt

# - - - OpenCV

# turns the given list of alternating x and y values into array with array of
#  points for, e.g., fillPoly() or polylines()
def xy2Pts(*xys):
  return np.int32(np.array(xys).reshape(-1, 2)[None] + .5)

# returns image; if color value is integer, it is used for all channels
# note: cv2.getTextSize() uses order w, h ("x first")
def getImg(h, w, nCh=3, color=255):
  img = np.zeros((h, w, nCh) if nCh > 1 else (h, w), np.uint8)
  if isinstance(color, tuple):
    img[:,:] = color
  else:
    img[...] = color
  return img

# shows image, possibly resizing window
def imshow(winName, img, resizeFctr=None, maxH=1000):
  h, w = img.shape[:2]
  h1 = None
  if resizeFctr is None and h > maxH:
    resizeFctr = float(maxH)/h
  if resizeFctr is not None:
    h1, w1 = intR(h*resizeFctr, w*resizeFctr)
    if MAC:
      img = cv2.resize(img, (0,0), fx=resizeFctr, fy=resizeFctr)
    else:
      cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
  cv2.imshow(winName, img)
  if h1 and not MAC:
    cv2.resizeWindow(winName, w1, h1)

# min max normalizes the given image
def normalize(img, min=0, max=255):
  return cv2.normalize(img, None, min, max, cv2.NORM_MINMAX, -1)

# shows normalized image
def showNormImg(winName, img):
  imshow(winName, normalize(img, max=1))

# converts the given image to gray
def toGray(img):
  return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# converts the given image to color (BGR)
# note: possibly include conversion to np.uint8
def toColor(img):
  return img if numChannels(img) > 1 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# returns whether the given image is grayscale
def isGray(img):
  return numChannels(img) == 1 or \
    np.array_equal(img[:,:,0], img[:,:,1]) and \
    np.array_equal(img[:,:,0], img[:,:,2])

# returns rectangular subimage, allowing tl and br points outside the image
# note: tl is included, br is excluded
def subimage(img, tl, br):
  h, w = img.shape[:2]
  tlx, tly = tl
  brx, bry = br
  return img[max(tly,0):min(bry,h), max(tlx,0):min(brx,w)]

# returns the image overlayed with the given mask in the given color
#  (mask values: 0: keep image ... 255: use color)
def overlay(img, mask, color=COL_G):
  imgC = pilImage(toColor(img))
  if mask.dtype != np.uint8:
    mask = np.rint(mask).astype(np.uint8)
  imgC.paste(color[::-1], None, pilImage(mask))
  return cvImage(imgC)

# extends the given image
# note: possibly easier: use cv2.copyMakeBorder()
def extendImg(img, trbl, color=255):
  h, w = img.shape[:2]
  x, y = trbl[3], trbl[0]
  imgE = getImg(y + h + trbl[2], x + w + trbl[1], nCh=numChannels(img),
    color=color)
  imgE[y:y+h, x:x+w] = img
  return imgE

# returns the bottom right point of the given image
def bottomRight(img):
  return (img.shape[1]-1, img.shape[0]-1)

# returns image rotated by the given angle (in degrees, counterclockwise)
def rotateImg(img, angle):
  cntr = tupleMul(img.shape[:2], .5)
  mat = cv2.getRotationMatrix2D(cntr, angle, 1.)
  return cv2.warpAffine(img, mat, img.shape[:2], flags=cv2.INTER_LINEAR)

# returns the number of channels in the given image
def numChannels(img): return img.shape[2] if img.ndim > 2 else 1

# returns the image size as tuple (width, height)
def imgSize(img): return img.shape[1::-1]

# creates large image out of small ones
# params: imgs: list of images or of (image, header text) tuples
#  nc: number columns, d: distance between small images,
#  hdrs: header text, style: for text, hd: header distance,
#  adjustHS: whether to adjust horizontal spacing to fit headers
# returns: large image, list of tuples with positions (tl) of small images
def combineImgs(imgs, nc=10, d=10, hdrs=None, style=None, hd=3, adjustHS=True,
    resizeFctr=None):
  if isinstance(imgs[0], tuple):
    assert hdrs is None
    imgs, hdrs = zip(*imgs)
  style = style or textStyle()
  if resizeFctr is not None:
    imgs = [cv2.resize(img, (0,0), fx=resizeFctr, fy=resizeFctr)
      for img in imgs]
  nnimgs = [img for img in imgs if img is not None]
  h, w = (max(img.shape[i] for img in nnimgs) for i in (0, 1))
  nCh = max(numChannels(img) for img in nnimgs)
  hdrH = textSize(hdrs[0], style)[1] if hdrs else 0
  if not hdrs: hd = 0
  wA = max(w, max(textSize(hdr, style)[0] for hdr in hdrs)) \
    if adjustHS and hdrs else w
  nr, nc = math.ceil(float(len(imgs))/nc), min(len(imgs), nc)
  hL, wL = nr*(h+d+hdrH+hd), nc*(wA+d)
  imgL = getImg(hL, wL, nCh)
  xyL, ndL = len(imgs)*[None], imgL.ndim
  for i, img in enumerate(imgs):
    r, c = divmod(i, nc)
    xL, yL = d/2 + c*(wA+d), d/2+hdrH+hd + r*(h+d+hdrH+hd)
    if img is not None:
      h1, w1 = img.shape[:2]
      imgL[yL:yL+h1, xL:xL+w1] = img if img.ndim == ndL else img[..., None]
    if hdrs and hdrs[i]:
      putText(imgL, hdrs[i], (xL, yL-hd), (0,0), style)
    xyL[i] = (xL, yL)
  return imgL, xyL

# returns the median of the given image
def median(img):
  return np.median(img.copy())

# returns Canny edge image and fraction of non-black pixels (before post blur)
def edgeImg(img, thr1=20, thr2=100, preBlur=True, postBlur=True, preNorm=False):
  img = toGray(img)
  if preNorm:
    img = normalize(img)
  if preBlur:
    img = cv2.GaussianBlur(img, (3, 3), 0)
  img = cv2.Canny(img, thr1, thr2)
  nz, npx = img.nonzero()[0].size, img.shape[0]*img.shape[1]
  if postBlur:
    img = cv2.GaussianBlur(img, (3, 3), 0)
  return img, float(nz)/npx

# matches the given template(s) against the given image(s)
#  e.g., (img, tmpl, img2, tmpl2, 0.5)
#    note: second match is weighted with factor 0.5 (default: 1)
# returns result image, top left x, top left y, bottom right (as tuple),
#  minimum distance between template and image border, match value,
#  and non-normalized match values
def matchTemplate(img, tmpl, *args):
  imgs, tmpls, fctrs = [img], [tmpl], [1]
  idx = 0
  for arg in args:
    if isinstance(arg, (int,float)):
      fctrs[idx] = arg
    else:
      if idx > len(tmpls)-1:
        tmpls.append(arg)
      else:
        imgs.append(arg)
        fctrs.append(1)
        idx += 1
  res, maxVals = 0, []
  for i, t, f in zip(imgs, tmpls, fctrs):
    r = cv2.matchTemplate(i, t, cv2.TM_CCOEFF_NORMED)
    maxVals.append(cv2.minMaxLoc(r)[1])
    if len(imgs) > 1:
      r = normalize(r, max=1) * f
    res += r
  minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(res)
  tlx, tly = maxLoc
  br = (tlx+tmpl.shape[1], tly+tmpl.shape[0])
  minD = min(min(maxLoc), img.shape[1]-br[0], img.shape[0]-br[1])
  return res, tlx, tly, br, minD, maxVal, maxVals

# returns the normalized correlation of the given images
# note: values range from -1 to 1 (for identical images)
def normCorr(img1, img2):
  assert img1.shape == img2.shape
  r = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
  assert r.shape == (1,1)
  return r[0,0]

# returns the default text style
def textStyle(size=.9, color=COL_BK):
  return (cv2.FONT_HERSHEY_PLAIN, size, color, 1, cv2.CV_AA)

# returns tuple with text width, height, and baseline; style is list with
#  putText() args fontFace, fontScale, color, thickness, ...
# note: 'o', 'g', and 'l' are identical in height and baseline
def textSize(txt, style):
  wh, baseline = cv2.getTextSize(txt, *[style[i] for i in [0,1,3]])
  return wh + (baseline,)

# puts the given text on the given image; whAdjust adjusts the text position
#  using text width and height (e.g., (-1, 0) subtracts the width)
# note: pos gives bottom left corner of text
def putText(img, txt, pos, whAdjust, style):
  adj = tupleMul(whAdjust, textSize(txt, style)[:2])
  cv2.putText(img, txt, intR(tupleAdd(pos, adj)), *style)

# returns new VideoCapture given filename or device number and checks whether
#  the constructor succeeded
def videoCapture(fnDev):
  cap = cv2.VideoCapture(fnDev)
  if not cap.isOpened():
    raise VideoError("could not open VideoCapture for %s" %fnDev)
  return cap
# returns FOURCC for the given VideoCapture (e.g., 'MJPG')
def fourcc(cap):
  fcc = int(cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
  return "".join(chr(fcc >> s & 0xff) for s in range(0,32,8))
# returns frame rate in fps for the given VideoCapture (e.g., 7.5)
def frameRate(cap):
  return float(cap.get(cv2.cv.CV_CAP_PROP_FPS))
# returns the frame count for the given VideoCapture
def frameCount(cap):
  return int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
# returns the given frame for the given VideoCapture
def readFrame(cap, n):
  setPosFrame(cap, n)
  ret, frm = cap.read()
  if not ret:
    raise VideoError("no frame %d" %n)
  return frm
# sets the current position for the given VideoCapture
def setPosFrame(cap, n=0):
  cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, n)

# returns PIL image given OpenCV image
def pilImage(img):
  return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if
    numChannels(img) > 1 else img)

# returns OpenCV image given PIL image
def cvImage(img):
  if isinstance(img, (Image.Image, ImageFile.ImageFile)):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  raise ArgumentError("not implemented for %s" %type(img))

# - - - Qt

# returns QImage given OpenCV image
def qimage(img):
  return array2qimage(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# - - -

if __name__ == "__main__":
  print "testing"
  _time2strTest()
  _time2sTest()
  _replaceCheckTest()
  _multiMatchTest()
  _beforeAfterTest()
  _invertTest()
  print "done"


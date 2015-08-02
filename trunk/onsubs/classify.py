#
# classify "on"/"off" substrate for all frames in video
#
# 6 May 2014 by Ulrich Stern
#

import argparse, os, shutil, numpy as np

from util import *
from common import *

# - - -

DEBUG = False

LOG_FILE, CLASSIFICATION_FILE = "__classify.log", "__classify.data"

PRED_DATA_DIR = '/dev/shm/tmp-for-pred/'
  # notes:
  # * overwritten by this script
  # * each GPU has its own subdirectory (e.g., gpu0)
  # * total amount of data for 8h video assuming flies are "inSap:"
  #  216k*2 / 600 * 5MB (batch with 600 92x92 images) = 720 * 5MB = 3.6GB
  # * data per GPU subdirectory:
  #  36 batches * 5MB/batch = 180MB
  # * uses "ram disk" (previously: DATA_DIR+'cc-exp2/tmp-for-pred/')
PRED_SUBDIR = 'pred/'
CC_DIR = ccDir(dropOut=False)

# - - -

def parseOptions():
  global opts
  p = argparse.ArgumentParser(
    description='Classify "on"/"off" substrate for all frames in video.')

  video, expDir = '1', '2014-03-23__21-05-44'
  p.add_argument('-v', dest='video', default=video, metavar='V',
    help='name of video file or video number (default: %(default)s);' +
      ' can be comma-separated list')
  addGpuOption(p)
  p.add_argument('-d', dest='expDir', default=expDir, metavar='D',
    help='experiment directory with models to use (default: %(default)s)')
  p.add_argument('--fr', dest='frameRange', default=None, metavar='R',
    help='limit frame range (e.g., "1000-1050" or "1000+50")')
  p.add_argument('--vd', dest='videoDir', action='store_true',
    help='store classification in video directory (instead of in %s)'
      %CLASSIFICATION_FILE)
  p.add_argument('--show', dest='show', action='store_true',
    help='show video while classifying')

  opts = p.parse_args()
  overrideOptions()

def videos(): return opts.video.split(',')

def overrideOptions():
  o = Overrider(opts)
  if len(videos()) > 1:
    o.override("videoDir")
  o.report()

# - - - shared with analyze.py

def frameRange_(tlen, d):
  if d is not None:
    fr = d.get('frameRange')
    return (0, tlen) if fr is None else fr
  if opts.frameRange:
    fr = re.findall(r'^(\d+)[+-](\d+)$', opts.frameRange)
    if not fr:
      error('frame range "%s" not recognized' %opts.frameRange)
    fr = map(int, fr[0])
    if '+' in opts.frameRange:
      fr[1] += fr[0]
    if fr[0] >= fr[1]:
      error('frame range start must be less than end')
    return fr[0], min(fr[1], tlen)
  else:
    return 0, tlen

def frameRange(tlen, d=None):
  fr = frameRange_(tlen, d)
  print "  frame range: %d-%d" %fr
  return fr

def printVideoModels(fv, ep):
  vn = fv.videoNum()
  print "  fly video: %s" %(fv.filename() if vn is None else vn)
  print "  models: %s" %basename(ep)

# returns next classification file (naming scheme: "video class-1.data") or
#  the maximum classification file number
def classificationFile(fv, maxCn=False):
  fp, fn = os.path.split(fv.filename(" class"))
  nums = multiMatch(r'^%s-(\d+)\.data$' %fn, os.listdir(fp))
  maxN = max(int(n) for n in nums) if nums else 0
  return maxN if maxCn else os.path.join(fp, "%s-%d.data" %(fn, maxN+1))

# - - -

def predDataDir():
  return PRED_DATA_DIR + 'gpu%d/' %(opts.gpu if opts.gpu else 0)

def dataDirPrep(dtDir):
  pdd = predDataDir()
  if os.path.isdir(pdd):
    shutil.rmtree(pdd)
  shutil.copytree(dtDir, pdd)

def updateBatchesMeta(dataDir, minMax, minB, maxB):
  assert len(minB) == len(maxB) == len(minMax)
  if len(minB) == 0:
    return
  minAll, maxAll = min(minB), max(maxB)
  if DEBUG:
    mis, mas = [], []
    for mmB in minMax:
      for mm in mmB:
        if mm != (0, 0):
          mis.append(mm[0]), mas.append(mm[1])
    print "\n  minAll, maxAll: %d, %d -- calc: %d, %d" \
      %(minAll, maxAll, min(mis), max(mas))
  fn = bmFn(dataDir)
  bm = unpickle(fn)
  bm['YL_min_distance_black_white'] = min(minAll, 255-maxAll)
  bm['YL_min_max_all'] = (minAll, maxAll)
  bm['YL_min_max'][:len(minMax)] = minMax
  pickle(bm, fn)

# returns Pr("on") for each image
def predictBatches(ep, nb, bSz):
  mds = [d for d in sorted(os.listdir(ep)) if \
    os.path.isdir(ep+d) and d.startswith(CONVNET_PREFIX)]
  pdd = predDataDir()
  with open(LOG_FILE, 'w', 1) as lf:
    predsSum, cmd0 = np.zeros((bSz*nb, 2), np.float32), None
    for i, md in enumerate(mds):   # for each model
      printF('\r  model %d' %(i+1))
      cmd = [SHOWNET, "-f", ep+md,
        "--write-features=probs", "--feature-path="+pdd+PRED_SUBDIR,
        "--data-path="+pdd,
        "--multiview-test=1", "--mirror-test=0",
        "--multirotate-test=0", "--multinorm-test=1",
        "--test-range=1-%d"%nb]
      execute(addGpu(cmd, opts), CC_DIR, lf)
      if i == 0:
        cmd0 = list(cmd)

      for j in range(nb):
        lbls, preds = readPredictionBatch(pdd+PRED_SUBDIR, j+1)
        if i == 0 and j == 0:
          dtMult = lbls.size / bSz
        assert bSz * dtMult == lbls.size
        predsSum[j*bSz:(j+1)*bSz, :] += multiviewAverage(lbls, preds, dtMult)[1]

    return predsSum[:, 1] / predsSum.sum(axis=1), dtMult, cmd0

# classify fly video
def classifyVideo(fv, ep):
  dtDir, db = dataDirBatch(ep)
  dataDirPrep(dtDir)
  fis, bSz = db.imgSize(), db.batchSize()
  print "  batch size: %d" %bSz
  bNums = range(1,3) if DEBUG else db.batchNums()
  assert bNums == range(1, len(bNums)+1)
  tlen, fisH, fisS, inSap = fv.tlen(), fis/2, fis**2, fv.inSapKeys()
  fr = frameRange(tlen)

  batch, labels = np.zeros((fisS, bSz), np.uint8), bSz*[0]
  if opts.show:
    imgs, hdrs = 2*[None], 2*[None]
  bIdx, idx, bc, predict, keys, allPreds = 0, 0, 1, False, [], []
  minMax, minB, maxB = [], [], []

  def writeBatch(nimgs):
    printF('\r  writing batch %d' %bc)
    pickle({'data': batch, 'labels': labels}, dbFn(bIdx+1, predDataDir()))
    mi, ma = np.amin(batch, axis=0), np.amax(batch, axis=0)
    minMax.append(zip(mi, ma))
    minB.append(min(mi[:nimgs])), maxB.append(max(ma[:nimgs]))

  print "\nprocessing:"
  t, lastKey, cmd = Timer(useClock=False), (1, fr[1]-1), None
  for fi in xrange(*fr):
    for f in [0, 1]:
      key = (f, fi)

      if key in inSap:
        fimg = fv.getFlyImage(key, fisH, channel=1)[0]
        batch[:,idx] = fimg.reshape(fisS)
        keys.append(key)
        idx = (idx + 1) % bSz
        if idx == 0:   # done with batch
          writeBatch(bSz)
          bIdx, bc = (bIdx + 1) % len(bNums), bc + 1
          if bIdx == 0:   # done with batch set
            predict = True
        if opts.show:
          imgs[f], hdrs[f] = fimg, str(fi)
          cv2.imshow("video", combineImgs(imgs, hdrs=hdrs)[0])
          cv2.waitKey(1)

      if key == lastKey:
        if idx != 0:
          writeBatch(idx)
          bIdx += 1
        predict = True

      if predict and keys:
        nb = len(bNums) if bIdx == 0 else bIdx
        print "\r  wrote %d batch%s in %.1fs (at frame %d)" \
          %(nb, "" if nb == 1 else "es", t.get(), fi)
        updateBatchesMeta(predDataDir(), minMax, minB, maxB)
        preds, dtMult, cmd = predictBatches(ep, nb, bSz)
        print "\r  predictions took %.1fs%s" \
          %(t.get(), "" if allPreds else " (data multiplier: %d)" %dtMult)
        allPreds.extend(zip(keys, preds[:len(keys)]))
        predict, keys, minMax, minB, maxB = False, [], [], [], []

  cf = classificationFile(fv) if opts.videoDir else CLASSIFICATION_FILE
  pickle({'allPreds': allPreds, 'videofn': fv.filename(), 'expPath': ep,
      'frameRange': fr, 'cmd': cmd},
    cf, backup=True)
  return len(allPreds), cf

def classify():
  for i, v in enumerate(videos()):
    if i: print
    classify_(v)

def classify_(v):
  print "=== classifying ==="
  if re.match(DIGITS_ONLY, v):
    v = int(v)
    if v >= len(FlyVideo.FNS):
      error("no video with number %d" %v)
    fv = FlyVideo(vn=v)
  else:
    checkIsfile(v)
    fv = FlyVideo(fn=v)

  ep = expPath(expandExpDir(opts.expDir))
  printVideoModels(fv, ep)

  t = Timer(useClock=False)
  np, cf = classifyVideo(fv, ep)
  print "done"
  print "\ntotal time: %.1fs" %t.get()
  print "number of predictions: %d" %np
  print "wrote classification to: %s" %os.path.basename(cf)

# - - -

if __name__ == "__main__":
  parseOptions()
  classify()


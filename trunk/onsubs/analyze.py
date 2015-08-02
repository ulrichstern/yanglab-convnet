#
# analyze "on"/"off" substrate classification of video
#
# 6 May 2014 by Ulrich Stern
# (split off from classify.py on 16 Aug 2014)
#
# notes:
# * analysis can handle both "off" substrate and "wrong side" eggs
# * cmd for sucrose paper:
#  python analyze.py -f __classify.data.v1,8,9,10
#    --wVImgs --bes 360 --priorEgg --sfx .4 --useReg
# * cmds for net paper:
#  videos: 1,8,9,10,16,17,18,19
#  analyze.py -f 1 --wImgs --ef --inVis   # "problem" images
#  analyze.py -f 1,8,9,10 --wVImgs --cntrls --bes 20 [--allEggs]
#    note: --allEggs can cause overlap
#  analyze.py -f 1,8,9,10 --lastVis --lbls sucrose,- [--absPI .75]
#  analyze.py -f 20 --wVImgs --cntrls --bes 20 --ctrlSub B --eggsOn B   # NPF
#

import argparse, os, sys, operator, collections
import numpy as np, numpy.random as nr

from util import *
from common import *
import classify

# - - -

FPS = 7.5   # note: would be better to get this from video

BEFORE_EGG_S, NUM_CONTROLS = 60, 40
CLASS_IMG_DIR, VISIT_IMG_DIR = "class-imgs/", "visit-imgs/"
VISIT_FILE = "visits.csv"
SINGLE_UPPERCASE = re.compile(r'([A-Z])')
VID_NUM_CLASS_NUM = re.compile(r'^\d+(-\d+)?$')

TXT_STL_EGG_TIME = (cv2.FONT_HERSHEY_PLAIN, 0.8, COL_BK, 1, cv2.CV_AA)
COL_SUCROSE, COL_PLAIN = COL_R, COL_G_D
COL_WITH_EGG, COL_EGG = COL_B_D, COL_Y

OP_TIME, OP_FRM_NUM, OP_NEITHER = 'T', 'FN', 'N'
OPTS_FN_N = (OP_FRM_NUM, OP_NEITHER)
OP_TOP, OP_BOTTOM = 'T', 'B'
OPTS_TB = (OP_TOP, OP_BOTTOM)

# - - -

def parseOptions():
  global opts
  p = argparse.ArgumentParser(
    description='Analyze "on"/"off" substrate classification of video.')

  g = p.add_argument_group('shared options')
  g.add_argument('-f', dest='file', metavar='F/N',
    nargs='?', const=classify.CLASSIFICATION_FILE, default=None,
    help='analyze the given file (default: %(const)s) or video number; ' +
      'can be comma-separated list; classification file number can be ' +
      'given after video number (e.g., "1-2")')
  g.add_argument('--cmd', dest='showCmd', action='store_true',
    help='show cuda-convnet command that was used for predictions')
  g.add_argument('--ef', dest='eggFile', metavar='F',
    nargs='?', const=FlyVideo.EGG_FILE, default=None,
    help='use the given egg file (default: %(const)s)')
  g.add_argument('--allEggs', dest='allEggs', action='store_true',
    help='analyze all eggs (default: only class 1); can cause overlap')
  g.add_argument('--shEt', dest='showEggTimes', action='store_true',
    help='show egg times')
  g.add_argument('--bes', dest='beforeEggS',
    type=int, default=BEFORE_EGG_S, metavar='N',
    help='number of seconds to analyze before each egg (default: %(default)s)')
  g.add_argument('--endJBV', dest='endJustBeforeVisit', action='store_true',
    help='analysis window ends just before egg-laying visit ' +
      '(instead of egg-laying)')
  g.add_argument('--useReg', dest='useRegions', action='store_true',
    help='use regions (top, middle, bottom) instead of classification')

  contextSize, flagSeqLenTh = 3, 3
  g = p.add_argument_group('"problem" images')
  g.add_argument('--shImgs', dest='showImages', action='store_true',
    help='show "problem" (e.g., flagged) images')
  g.add_argument('--wImgs', dest='writeImages', action='store_true',
    help='write "problem" images to %s (which suppresses showing them)'
      %CLASS_IMG_DIR)
  g.add_argument('--ba', dest='contextSize',
    type=int, default=contextSize, metavar='N',
    help='number of images to show before/after "problem" images ' +
      '(default: %(default)s)')
  g.add_argument('--sl', dest='flagSeqLenTh',
    type=int, default=flagSeqLenTh, metavar='N',
    help='flag sequences for which the class changes for less than or ' +
      'equal to the given number of frames (default: %(default)s)')
  g.add_argument('--exCh', dest='excludeChanges', action='store_true',
    help='exclude images whose class is changed by filter')
  g.add_argument('--inVis', dest='includeVisits', action='store_true',
    help='include images at start and end of substrate visits')
  g.add_argument('--tmFn', dest='showTimesFrameNumbers',
    default=OP_TIME, choices=OPTS_FN_N,
    help='show frame numbers instead of times (%s) or neither (%s)' %OPTS_FN_N)

  absPI, topBottomLabels, allVisitsRowM = 1, 'top,bottom', 10
  g = p.add_argument_group('visits')
  g.add_argument('--alOl', dest='allowOverlap', action='store_true',
    help='allow overlap of analysis window with prior eggs')
  g.add_argument('--cntrls', dest='numControls', type=int, metavar='N',
    nargs='?', const=NUM_CONTROLS, default=0,
    help='compare eggs with the given number of controls (default: %(const)s)')
  g.add_argument('--ctrlSub', dest='controlsSub', default=None, choices=OPTS_TB,
    help='use top (%s) or bottom (%s) for controls' %OPTS_TB +
      ' (and skip preference index check)')
  g.add_argument('--absPI', dest='absPI',
    type=float, default=absPI, metavar='V',
    help='require the given absolute preference index when selecting the ' +
      'control substrate (e.g., 0.95; default: %(default)s)')
  g.add_argument('--wVis', dest='writeVisits', action='store_true',
    help='write details of substrate visits to %s' %VISIT_FILE)
  g.add_argument('--visTo', dest='visitsTo', default=None, choices=OPTS_TB,
    help='write stats for visits to top (%s) or bottom (%s) instead' \
      %OPTS_TB)
  g.add_argument('--lastVis', dest='lastVisit', action='store_true',
    help='write stats for last visit to top or bottom instead (default ' +
      'if no --visTo given: top)')
  g.add_argument('--lvIsMR', dest='lastVisitIsMR', action='store_true',
    help='write stats only for cases where last visit to top or bottom ' +
      'is the most recent visit')
  g.add_argument('--inLV', dest='includeLastVisit', action='store_true',
    help='include last (i.e., egg-laying) visit in "by egg stats"')
  g.add_argument('--lbls', dest='topBottomLabels', default=topBottomLabels,
    metavar='L',
    help='labels to use instead of "top" and "bottom" (comma-separated)')
  g.add_argument('--vVis', dest='visualizeVisits', action='store_true',
    help='visualize visits before eggs')
  g.add_argument('--priorEgg', dest='priorEgg', action='store_true',
    help='show only eggs where analysis window includes prior egg')
  g.add_argument('--eggsOn', dest='eggsOn', default=None, choices=OPTS_TB,
    help='show eggs on top only (%s) or bottom only (%s)' %OPTS_TB +
      ' (affects visualization only)')
  g.add_argument('--allV', dest='allVisitsRowM', type=int, metavar='N',
    nargs='?', const=allVisitsRowM, default=0,
    help='show all visits (not just visits before eggs) using the ' +
      'given number of minutes per row (default: %(const)s)')
  g.add_argument('--sfx', dest='scaleFactorX',
    type=float, default=0, metavar='F',
    help='scale factor for x dimension (e.g., 0.5 for "half size")')
  g.add_argument('--wVImgs', dest='writeVisitImgs', action='store_true',
    help='write visit images to %s' %VISIT_IMG_DIR)

  opts = p.parse_args()
  overrideOptions()

def overrideOptions():
  o = Overrider(opts)
  # note: order of overridden options matches options()
  if opts.beforeEggS != BEFORE_EGG_S or opts.endJustBeforeVisit or \
      opts.allowOverlap or opts.numControls or opts.includeVisits or \
      opts.writeVisits or opts.lastVisit or \
      opts.visualizeVisits or opts.priorEgg or opts.allVisitsRowM or \
      opts.writeVisitImgs:
    if opts.eggFile is None:
      o.override("eggFile", FlyVideo.EGG_FILE)
  if opts.lastVisit or opts.allVisitsRowM:
    # note: controls calculated for lastVisit, so allEggs required
    if not opts.allEggs:
      o.override("allEggs")
  if opts.numControls:
    if not opts.endJustBeforeVisit:
      o.override("endJustBeforeVisit", True, "window ends just before visit")
  if opts.writeImages:
    if not opts.showImages:
      o.override("showImages")
  if opts.priorEgg or opts.lastVisit:
    if not opts.allowOverlap:
      o.override("allowOverlap")
  if opts.lastVisit:
    if not opts.numControls:
      o.override("numControls", NUM_CONTROLS)
  if opts.visitsTo or opts.lastVisit:
    if not opts.writeVisits:
      o.override("writeVisits")
  if opts.lastVisit:
    if not opts.visitsTo:
      o.override("visitsTo", OP_TOP)
  if opts.writeVisitImgs or opts.allVisitsRowM:
    if not opts.visualizeVisits:
      o.override("visualizeVisits")
  if opts.allVisitsRowM:
    if not opts.scaleFactorX:
      o.override("scaleFactorX", .25)
  opts.scaleFactorX = opts.scaleFactorX or 1
  o.report()

# - - -

# TODO: use FlyVideo's version instead
def fi2time(fi, secs=True):
  return time2str(fi/FPS, '%H:%M:%S' if secs else '%H:%M', utc=True)

# returns analysis window size minus 1
def windowD(): return int(opts.beforeEggS*FPS)
def windowSz(): return windowD()+1

# returns, e.g., "f1 01:00:28"
def eggName(fv, f, fi): return "f%d %s" %(fv.flyNum(f), fi2time(fi))

# returns whether to consider eggs in top or bottom
# note: if not specified via opts.controlsSub, substrate with more eggs is
#  chosen while guaranteeing absolute preference index of opts.absPI for
#  each fly
def eggsInTop(fv, eggs):
  if opts.controlsSub:
    return opts.controlsSub == OP_TOP
  moreT = []
  for f, (t, b) in enumerate(fv.eggStats(eggs)):
    if t + b == 0:
      continue
    ff = " for fly %d" %fv.flyNum(f)
    if t == b:
      error("equal number of eggs on each side%s" %ff)
    api = abs(float(t-b)/(t+b))
    if api < opts.absPI:
      error("absolute preference index %.2f too small%s" %(api, ff))
    moreT.append(t > b)
  if len(moreT) == 2 and moreT[0] != moreT[1]:
    error("differing egg-laying preference for the two flies")
  return moreT[0]

def class1Eggs(fv, eggs, isOn):
  lf, inTop, cl1Eggs, nc1 = None, fv.inTop(), [], []
  for f, fi in eggs:
    if f != lf:   # new fly
      lf, cl1 = f, True
    else:
      et = inTop[f, fi]
      cl1 = np.any(np.logical_and(isOn[f, lfi:fi], inTop[f, lfi:fi] != et))
    if cl1:
      cl1Eggs.append((f, fi))
    else:
      nc1.append(fi2time(fi))
    lfi = fi
  print "  non-class 1 eggs: %s" \
    %((" ".join(nc1) if nc1 else "-") if opts.showEggTimes else len(nc1))
  return cl1Eggs

# returns list with keys of eggs and "is before egg" lookup table
def loadEggFile(fv, isOn):
  tlen, d = fv.tlen(), windowD()
  print
  eggs = fv.loadEggFile(opts.eggFile)
  c1es = class1Eggs(fv, eggs, isOn)
  if not opts.allEggs:
    print "    keeping only class 1 eggs"
    eggs = c1es
  isBe = np.zeros((2, tlen), np.uint8)
  for f, fi in eggs:
    isBe[f, max(0, fi-d):min(tlen, fi+1)] = 1
  if opts.showEggTimes and eggs:
    print "  times: %s" %" ".join(fi2time(fi) for f, fi in eggs)
  return eggs, isBe

# if video contributed to test set, calculate error rate
def checkTestSet(ep, fv, isOn, fr):
  db, vn = dataDirBatch(ep)[1], fv.videoNum()
  tbns, bSz = map(int, ccOpts(ep)['test-range'].split(',')), db.batchSize()
  print "\nerror rate on test set:"
  nerrs = ntest = 0
  bnUsed = []
  for bn in tbns:
    if db.vnKey(bn, 0)[0] == vn:
      bnUsed.append(bn)
      for i in xrange(bSz):
        f, fi = db.vnKey(bn, i)[1]
        if fr[0] <= fi and fi < fr[1]:
          nerrs += isOn[f, fi] != db.onSubstrate(bn, i)
          ntest += 1
  if not bnUsed:
    print "  fly video did not contribute to test set"
    return
  print "  batch size: %d   batch numbers used: %s" %(bSz, join(", ", bnUsed))
  print "  tests images: %d   errors: %d   error rate: %s" \
    %(ntest, nerrs, "-" if ntest == 0 else "{:.3%}".format(nerrs/float(ntest)))

# replaces value with majority "vote" for kernel of size ksize around frame
# notes:
# * for binary values, median filter is equivalent to majority filter
# * OpenCV's medianBlur() extends edge values into the border, see test
def majorityFilter(isOn, ksize=5, verbose=False):
  if verbose:
    print "\napplying majority filter (ksize=%d)..." %ksize
  def fltr(a): return cv2.medianBlur(a, ksize).squeeze()
  isOnF = np.apply_along_axis(fltr, axis=1, arr=isOn)
  chKeys = set(zip(*(isOn != isOnF).nonzero()))
  if verbose:
    print "  classification changes: %d" %len(chKeys)
  return isOnF, chKeys

def _majorityFilterTest():
  def nda(a): return np.array(a, np.uint8).reshape(1,-1)
  def mf(*args): return majorityFilter(*args)[0]
  ts = [([nda([0,1,0,1,0,0,0,1,1,0])], [0,0,0,0,0,0,0,0,0,0]),
    ([nda([1,0,1,0,1,1,1,0,0,1])], [1,1,1,1,1,1,1,1,1,1])]
  for args, r in ts:
    test(mf, args, r)

def showProblemImages(fv, isOn, prOn, flagged, chKeys, eggs, isBe, vis, fr):
  insbe = "" if isBe is None else "in %ds before eggs" %opts.beforeEggS
  if opts.showImages:
    print '\n%s "problem" images' \
      %("writing" if opts.writeImages else "showing")
  else:
    nf = sum(isBe[k] for k in flagged.keys()) if insbe else len(flagged)
    print "\nnumber of flagged images%s: %d" %(' '+insbe if insbe else '', nf)
    return
  print "  labels that were changed by filter are in yellow"
  if insbe:
    print "  showing only images %s" %insbe
  txtStl = list(TXT_STL_ON); txtStl[2] = COL_Y

  # get "context"
  fisH, key2img, tlen, cs = 72/2, {}, isOn.shape[1], opts.contextSize
  toShw = set(flagged.keys()) | (set() if opts.excludeChanges else chKeys)
  visStrt = {}
  if opts.includeVisits:
    for f, ei, fi, li, wfli, top in vis:
      toShw.update([(f, fi), (f, li)])
      visStrt[(f, fi+cs)] = ei
  for f, fi in sorted(toShw, key=operator.itemgetter(1)):
    for o in range(-cs, cs+1):
      if 0 <= fi+o and fi+o < tlen:
        key = (f, fi+o)
        key2img[key] = fv.getFlyImage(key, fisH)[0]

  imgs, inSeq, nr, nc, eggs = [], False, 10, 16, set(eggs)
  niTh, lastFi = (nr-1)*nc, fr[1]-1
  for f in [0, 1]:
    imn, fn = 1, fv.flyNum(f)
    for fi in xrange(*fr):
      key = (f, fi)
      img = key2img.get(key)
      if isBe is not None and not isBe[key]:
        img = None
      if img is not None:
        if not inSeq and len(imgs) > 0:
          imgs.append((None, ''))
        po = prOn.get(key, -1)
        labelImg(img, isOn[key], None if (po > .5) == isOn[key] else txtStl)
        # note: to hide majority filter: labelImg(img, po > .5)
        if not inSeq and opts.showTimesFrameNumbers != OP_NEITHER:
          hdr = fi2time(fi) if opts.showTimesFrameNumbers == OP_TIME \
            else str(fi)
        else:
          hdr = '-' if po < 0 else '%.0f' %(po*100)
          if key in eggs: hdr += ' egg'
          elif key in visStrt: hdr += ' e%d' %visStrt[key]
          elif key in flagged:
            fl = flagged[key]
            hdr += ' **' if fl == -1 else ' *%d*' %fl
        imgs.append((img, hdr))
        inSeq = True
      if img is None and inSeq and len(imgs) > niTh or fi == lastFi:
        if imgs:
          imgL = combineImgs(imgs, nc=nc)[0]
          if opts.writeImages:
            cv2.imwrite(CLASS_IMG_DIR+"v%d f%d img %d.png" \
              %(fv.videoNum(), fn, imn), imgL)
            imn += 1
          else:
            cv2.imshow("problem images (fly %d)" %fn, imgL)
            cv2.waitKey(0)
        imgs, inSeq = [], False
      elif img is None and inSeq:
        inSeq = False

# - - - "visit level" analysis

def fnosTxt(fv, f, fi):
  return "egg %s (%d): fly not on substrate" %(eggName(fv, f, fi), fi)

# returns start frame of egg-laying visit (None for "off substrate" egg)
#  and egg index for each egg
def egg2vfi(fv, isOn, eggs):
  e2vfi = {}
  for ei, key in enumerate(eggs):
    (f, fi), inT = key, fv.inTop(*key)
    if not isOn[key]:
      e2vfi[key] = (None, ei)
      continue
    for fib in xrange(fi-1, -1, -1):
      if not isOn[f, fib] or fv.inTop(f, fib) != inT:
        e2vfi[key] = (fib + 1, ei)
        break
  return e2vfi

# allVis entry: f, ei, fi, li, top
# notes:
# * ei None for visit without egg-laying
# * for visit with multiple eggs, minimum ei is used (earliest egg)
# * sorted by (f, fi)
def allVisits(fv, isOn, eggs):
  allVis, e2vfi, vfi2ei = [], egg2vfi(fv, isOn, eggs), {}
  for (f, fi), (vfi, ei) in e2vfi.iteritems():
    if vfi is not None:
      ei1 = vfi2ei.get((f, vfi))
      vfi2ei[(f, vfi)] = ei if ei1 is None else min(ei, ei1)
  isOn1 = np.concatenate((isOn, np.zeros((2, 1), np.bool)), axis=1)
  isOnCh = sorted(zip(*(isOn1[:, 1:] != isOn1[:, :-1]).nonzero()))
    # cases where isOn is different for next frame
    # notes:
    # * sorted() seems not needed but no guarantee in nonzero() docs
    # * does not include unlikely events of jumps between the two substrates
    #  (which can cause "assert not vfi2ei" to fail)
  for i, (f, fi) in enumerate(isOnCh):
    if isOn[f, fi]:   # visit end
      assert not isOn1[f, fi+1]
      firstCh = i == 0 or f != isOnCh[i-1][0]   # first change for fly
      vfi = 0 if firstCh else isOnCh[i-1][1] + 1
      ei = vfi2ei.get((f, vfi))
      if ei is not None:
        del vfi2ei[(f, vfi)]
      allVis.append((f, ei, vfi, fi, fv.inTop(f, fi)))
  assert not vfi2ei
  return allVis

def addVisits(fv, isOn, vis, wfi, wli, f, ei):
  vtb = vfi = None   # vtb: in visit (None, top: True, bottom: False)
  for fib in xrange(wfi,wli+1):
    io, inT = isOn[f, fib], fv.inTop(f, fib)
    if io and vtb != inT:   # visit start
      if vtb is not None:   # also end
        vis.append((f, ei, vfi, fib-1, (wfi, wli), vtb))
      vtb, vfi = inT, fib
    if vtb is not None and (not io or fib == wli):   # visit end
      vis.append((f, ei, vfi, fib-1 if not io else fib, (wfi, wli), vtb))
      vtb = vfi = None

def reportSkipped(skipped, indent=True):
  ind = "  " if indent else ""
  print "%sskipped eggs:%s" %(ind, "" if skipped else " none")
  if skipped:
    for k, l in sorted(skipped.iteritems()):
      print "%s  %s: %d%s" %(ind, k, len(l), " (%s)" %", ".join(l) if l else "")

# vis entry (visit): f, ei, fi, li, (wfi, wli), top  (wli: window last index)
def analyzeVisits(fv, isOn, eggs, isBe):
  vis, d, skipped = [], windowD(), collections.defaultdict(list)
  print "\nanalyzing visits"
  if not eggs:
    print "  skipped; requires egg file"
    return vis
  if opts.endJustBeforeVisit:
    e2vfi = egg2vfi(fv, isOn, eggs)
  for ei, (f, fi) in enumerate(eggs):
    def skippedAppend(k): skipped[k].append(eggName(fv, f, fi))
    nvis, firstEgg = len(vis), ei == 0 or f != eggs[ei-1][0]
    wli = e2vfi[(f, fi)][0] if opts.endJustBeforeVisit else fi
    if wli is None:
      skippedAppend('"off" substrate')
      continue
    wfi = wli-d
    if wfi < 0:
      skippedAppend('too early')
      continue
    elif not firstEgg and wfi <= eggs[ei-1][1]:
      if opts.allowOverlap:
        wfi = eggs[ei-1][1]
      else:
        skippedAppend('overlap')
        continue
    addVisits(fv, isOn, vis, wfi, wli, f, ei)
    if not isOn[f, fi]:
      print "  " + fnosTxt(fv, f, fi)
    elif not opts.endJustBeforeVisit:
      assert len(vis) > nvis and vis[-1][5] == fv.inTop(f, fi)
        # no egg without visit and last visit's top/bottom matches egg
  # check
  if not (opts.endJustBeforeVisit or skipped):
    isOn1 = np.zeros_like(isOn)
    for f, ei, fi, li, wfli, top in vis:
      isOn1[f, fi:li+1] = 1
    assert np.all(isOn1 == np.logical_and(isOn, isBe))

  print "  number visits: %d" %len(vis)
  reportSkipped(skipped)
  return vis

# returns indexes of eggs without prior egg in analysis window
def eggsWithoutPrior(eggs):
  noPrior, d = set(), windowD()
  for ei, (f, fi) in enumerate(eggs):
    if ei == 0 or f != eggs[ei-1][0] or fi-d > eggs[ei-1][1]:
      noPrior.add(ei)
  return noPrior

# returns "not before egg-laying" visits for both flies
# notes:
# * each "not before egg-laying" window should not overlap with
#  - the regular (before egg-laying) analysis windows
#  - the egg-laying visits
#  - other "not before egg-laying" windows
#  - the windows before and after "off substrate" eggs
# * vis matches analyzeVisits() but negative "egg indexes" used
def notBeforeEggVisits(fv, isOn, eggs):
  vis, nw, nei, rng = [], [0, 0], -1, nr.RandomState(42)
  excl = np.zeros_like(isOn)   # to look up li of window
  allVis, d = allVisits(fv, isOn, eggs), windowD()
  inT, nc = eggsInTop(fv, eggs), opts.numControls
  for f, ei, fi, li, top in allVis:
    if ei is not None:
      excl[f, fi-d:li+1+d] = True
  for f, fi in eggs:
    if not isOn[f, fi]:
      excl[f, fi-d:fi+1+d*2] = True
  for i in rng.permutation(len(allVis)):
    f, ei, fi, li, top = allVis[i]
    if top == inT and not excl[f, fi] and fi-d >= 0 and nw[f] < nc:
      assert ei is None
      addVisits(fv, isOn, vis, fi-d, fi, f, nei)
      excl[f, fi-d:fi+1+d] = True
      nw[f] += 1
      nei -= 1
      if sum(nw) == 2*nc:
        return vis
  print '  less than %d controls: %s' %(nc,
    ", ".join("fly %d: %d" %(fv.flyNum(f), nw[f])
      for f in (0, 1) if nw[f] < nc))
  return vis

def scl(x): return int(round(x*opts.scaleFactorX))
def drawBar(img, y, barh, fx, lx, top, withEgg=False):
  if y+barh > img.shape[0]:
    return
  img[y:y+barh, fx:lx+1] = COL_SUCROSE if top else COL_PLAIN
  if withEgg:
    img[y:y+(barh*2/3 if top else barh), fx:lx+1] = COL_WITH_EGG
def drawEgg(img, y, barh, x):
  pm = 4
  pts = xy2Pts(x-pm,y, x,y+barh-1, x+pm,y)
  cv2.fillPoly(img, pts, COL_EGG, lineType=cv2.CV_AA)
  cv2.polylines(img, pts, False, COL_BK, lineType=cv2.CV_AA)
def reportImagesWritten(ifns):
  if ifns:
    print "  wrote to %s: %s" %(VISIT_IMG_DIR, ", ".join(ifns))

def visualizeVisits(fv, vis, eggs):
  print "\nvisualizing visits"
  barh, barw, dy, dyel, dynbe, b = 10, scl(windowSz()), 3, (1,1), (12,5), 5
  tw, th = textSize(fi2time(0), TXT_STL_EGG_TIME)[:2]
  dtm, beh, nbeh = 5, barh+dy, dynbe[0]+th+dynbe[1]
  sclBarS = min(opts.beforeEggS, 60)   # sclBarS=0: none
  sbdy, sbh, sbtd = 12, 6, 3

  notShwn, ifns, noPrior, eggSet = [], [], eggsWithoutPrior(eggs), set(eggs)
  for f in [0, 1]:
    ne = sum(fl == f for fl, fi in eggs)
    if ne == 0:
      continue
    nbe = set(v[1] for v in vis if v[0] == f and v[1] < 0)
    nr, fn = ne + len(nbe), fv.flyNum(f)
    h = beh*nr-dy + 2*b + (sbdy+sbh+th+sbtd if sclBarS else 0) + \
      (nbeh if nbe else 0)
    img, cei, rw, yos = getImg(h, barw + dtm+tw + 2*b), None, -1, 0
    y = yos+b+beh*rw
    for fl, ei, fi, li, (wfi, wli), top in vis:
      if opts.eggsOn:
        if ei >= 0 and fv.inTop(*eggs[ei]) != (opts.eggsOn == OP_TOP):
          continue
      if fl != f or opts.priorEgg and ei in noPrior:
        continue
      def toX(x): return b+barw-1 - scl(wli-x)
      if ei != cei:   # new egg
        rw += 1
        if ei < 0 and yos == 0:
          yos = nbeh
          putText(img, 'not before egg-laying:',
            (b+(barw+dtm+tw)/2, b+beh*rw+dynbe[0]), (-.5, 1), TXT_STL_EGG_TIME)
        y, cei = yos+b+beh*rw, ei
        img[y+barh-1:y+barh, toX(wfi):b+barw, :] = 128
        putText(img, fi2time(eggs[ei][1] if ei >= 0 else wli),
          (b+barw+dtm, y), (0, 1), TXT_STL_EGG_TIME)
      drawBar(img, y, barh, toX(fi), toX(li), top)
      for fi1 in (fi, li):
        if (fl, fi1) in eggSet:
          img[y-dyel[0]:y+barh+dyel[1], toX(fi1):toX(fi1)+1] = COL_BK
    if sclBarS:
      x, y, w = b+barw-1, y+barh+sbdy, scl(sclBarS*FPS)
      img[y:y+sbh, x-w:x+1] = 0
      putText(img, '%ds'%sclBarS, (x-w/2, y+sbh+sbtd), (-.5, 1),
        TXT_STL_EGG_TIME)
    numNotShown = nr-rw-1
    if numNotShown:
      notShwn.append((fn, numNotShown))
      img = img[0:img.shape[0]-beh*numNotShown, ...]
    iname = "v%d f%d %ds" %(fv.videoNum(), fn, opts.beforeEggS)
    if opts.writeVisitImgs:
      ifns.append(iname+".png")
      cv2.imwrite(VISIT_IMG_DIR+ifns[-1], img)
    else:
      cv2.imshow(iname, img)
      cv2.waitKey(0)
  if notShwn:
    print "  skipped eggs (no visit %sin analysis window): %s" \
      %("or no prior egg " if opts.priorEgg else "",
        ", ".join("fly %d: %d" %e for e in notShwn))
  reportImagesWritten(ifns)

def visualizeAllVisits(fv, isOn, eggs):
  print "\nvisualizing all visits"
  barh, dy, b, rlen, tlen = 10, 3, 5, opts.allVisitsRowM*60*FPS, fv.tlen()
  nr, allVis, ifns = intR(float(tlen)/rlen), allVisits(fv, isOn, eggs), []
  tw = textSize(fi2time(0, secs=False), TXT_STL_EGG_TIME)[0]
  xoff = b+tw+5
  def toY(r): return (barh+dy)*r + b
  def toX(c): return xoff + scl(c)
  for fl in (0, 1):
    img = getImg((barh+dy)*nr + 2*b, scl(rlen) + xoff+b)
    for r in range(nr):
      y = toY(r) + barh-1
      img[y:y+1, xoff:xoff+scl(rlen)] = 128
      putText(img, fi2time(rlen*r, secs=False), (b, y), (0, 0),
        TXT_STL_EGG_TIME)
    for f, ei, fi, li, top in allVis:
      if f != fl:
        continue
      withEgg = ei is not None
      (r1, c1), (r2, c2) = divmod(fi, rlen), divmod(li, rlen)
      if r1 != r2:
        drawBar(img, toY(r1), barh, toX(c1), toX(rlen-1), top, withEgg)
        r1, c1 = r2, 0
      drawBar(img, toY(r1), barh, toX(c1), toX(c2), top, withEgg)
    for f, fi in eggs:
      if f == fl:
        re, ce = divmod(fi, rlen)
        drawEgg(img, toY(re), barh, toX(ce))
    iname = "v%d f%d" %(fv.videoNum(), fv.flyNum(fl))
    if opts.writeVisitImgs:
      ifns.append(iname+".png")
      cv2.imwrite(VISIT_IMG_DIR+ifns[-1], img)
    else:
      cv2.imshow(iname, img)
      cv2.waitKey(0)
  reportImagesWritten(ifns)

# - - - visits.csv
# header
# note: better to sort visits.csv-related functions by "type" of file

def writeStatsHeader(vf):
  if opts.writeVisits:
    vf.write('# command: %s\n\n' %' '.join(sys.argv))
    if opts.lastVisit:
      return writeLastVisitToStatsHeader(vf)
    elif opts.visitsTo:
      return writeVisitsToStatsHeader(vf)
    return writeByEggStatsHeader(vf)

def topBottomLabels():
  tbl = opts.topBottomLabels.split(',')
  if len(tbl) != 2:
    error('cannot parse labels "%s"' %opts.topBottomLabels)
  return tbl

def topBottomLabel(top, tbl=None):
  if tbl is None:
    tbl = topBottomLabels()
  elif isinstance(tbl, dict):
    tbl = tbl['tbl']
  return tbl[not top]

def writeByEggStatsHeader(vf):
  tl, bl = topBottomLabels()
  nv, avd = 'number visits', 'average visit duration'
  vfeCs = 'video,fly,egg number,'
  besCs = '%s %s,%s %s,%s %s,%s %s' %(nv, tl, nv, bl, avd, tl, avd, bl)
  vf.write('%svisit duration (seconds),visit to,%s'
    %(vfeCs, besCs))
  return dict(tbl=(tl, bl), vfeCs=vfeCs, besCs=besCs)

def writeVisitsToStatsHeader(vf):
  vtl = topBottomLabel(opts.visitsTo == OP_TOP)
  iwv = 'eggs/controls with visit to %s' %vtl
  vf.write('video,fly,egg vs. control,number of eggs/controls,' +
    'number of %s,percentage of %s\n' %(iwv, iwv))

def writeLastVisitToStatsHeader(vf):
  toTop, tbl = opts.visitsTo == OP_TOP, topBottomLabels()
  lbl = topBottomLabel(toTop, tbl)
  evc = 'egg vs. control'
  viw = 'visit to %s in %ds window' %(lbl, opts.beforeEggS)
  vf.write('video,fly,%s,egg/control time,' %evc +
    'time since last visit to %s (seconds),%s,' %(lbl, viw) +
    'whether last visit to %s is the most recent visit\n' %lbl)
  return dict(toTop=toTop, tbl=tbl,
    inW='%s,%s (V),no %s (NV),V+NV,V/(V+NV)' %(evc, viw, viw))

# - - -
# rows

def byEggStatsIdxs(vis):
  n = len(vis)
  idxs, lei = n*[False], None
  for v in xrange(n-1, -1, -1):
    ei = vis[v][1]
    if ei != lei:
      if opts.includeLastVisit:
        idxs[v] = True
      else:
        if v > 0 and vis[v-1][1] == ei:
          idxs[v-1] = True
    lei = ei
  return idxs

def writeByEggStats(vf, ds, bes, vfe):
  nv = [len(ds[top]) for top in (0, 1)]
  adb, adt = (sum(ds[top])/nv[top] if nv[top] else np.nan for top in (0, 1))
  s = (nv[1], nv[0], adt, adb)
  vf.write(',%d,%d,%.2f,%.2f' %s)
  bes.append((vfe, s))

def writeStatsForFile(vf, a, s, h):
  if opts.writeVisits:
    if opts.lastVisit:
      writeLastVisitToStatsForFile(vf, a, s, h)
    elif opts.visitsTo:
      writeVisitsToStatsForFile(vf, a, s, h)
    else:
      writeByEggStatsForFile(vf, a, s, h)

def writeByEggStatsForFile(vf, a, bes, h):
  fv, vis = a['fv'], a['vis']
  idxs, vn = byEggStatsIdxs(vis), fv.videoNum()
  lfei, lv = None, len(vis)-1
  for v, (f, ei, fi, li, wfli, top) in enumerate(vis):
    d, vfe = (li-fi+1)/FPS, [vn, fv.flyNum(f), ei]
    if lfei != (f, ei):   # new egg
      ds, lfei, lvfe = [[], []], (f, ei), vfe
    ds[top].append(d)
    vf.write('\n'+ join(',', vfe + ["%.2f" %d, topBottomLabel(top, h)]))
    if idxs[v]:
      writeByEggStats(vf, ds, bes, lvfe)

def eggVcontrol(egg): return "egg" if egg else "control"

def writeVisitsToStatsForFile(vf, a, s, h):
  fv, vis = a['fv'], a['vis']
  vn, top0 = fv.videoNum(), None
  w = {}   # (f, ei) -> number visits to visitsTo (possibly 0) in window
  for v, (f, ei, fi, li, (wfi, wli), top) in enumerate(vis):
    if li == wli:
      if top0 is None:
        top0 = top
      else:
        assert top == top0
    key = (f, ei)
    if key not in w:
      w[key] = 0
    if fv.inTop(f, fi) == (opts.visitsTo == OP_TOP):
      w[key] += 1
  for fl in (0, 1):
    for egg in (True, False):
      nall, nto = (sum(nv >= v for (f, ei), nv in w.iteritems()
        if f == fl and (ei >= 0 if egg else ei < 0))
        for v in (0, 1))
      if nall == 0:
        break
      vf.write(join(',', (vn, fv.flyNum(fl), eggVcontrol(egg),
        nall, nto, "{:.1%}".format(float(nto)/nall))) + '\n')

def offEggBeforeAfterVis(allVis, i, eggsS, tlen):
  f, ei, fi, li, top = allVis[i]
  (eb, ea), (pli, nfi) = beforeAfter(eggsS, (f,fi)), (-1, tlen)
  if i > 0:
    pv = allVis[i-1]
    if pv[0] == f:
      pli = pv[3]
  if i+1 < len(allVis):
    nv = allVis[i+1]
    if nv[0] == f:
      nfi = nv[2]
  return eb is not None and eb[0] == f and eb[1] > pli or \
    ea is not None and ea[0] == f and ea[1] < nfi

# notes:
# * the controls picked here are not restricted using the analysis "window"
#  since there is no window for "last visit to"
#  TODO: window now used, so possibly change
# * eggs are listed by iterating over allVis, causing, e.g., "off" substrate
#  eggs to be skipped
def writeLastVisitToStatsForFile(vf, a, s, h):
  fv, eggs, allVis, toTop = a['fv'], a['eggs'], a['allVis'], h['toTop']
  d, rng, tlen = windowD(), nr.RandomState(42), fv.tlen()
  eInT, eggsS = eggsInTop(fv, eggs), sorted(eggs)
  if s:
    inWin = s[0]
  else:
    inWin = dict((eggVcontrol(e), [0]*2) for e in (True, False))
    s.append(inWin)
  # returns time, int("whether in window"), whether last visit is most recent
  def tmSinceLstVisTo(allVis, i, toTop):
    f, j = allVis[i][0], i-1
    while j >= 0 and allVis[j][0] == f:
      if allVis[j][-1] == toTop:
        tm = allVis[i][2] - allVis[j][3]
        return "%.2f" %(tm / FPS), int(tm <= d), j == i-1
      j -= 1
    return (None, 0, False)
  def updateInWinCounts(eVc, inW): inWin[eVc][1-inW] += 1
  def checkSubstrate(eVc, top):
    if top != eInT:
      raise InternalError("wrong substrate for %s" %eVc)
  skipped = []
  for fl in (0, 1):
    vn, fn, ne, nc = fv.videoNum(), fv.flyNum(fl), 0, 0
    eVc = eggVcontrol(True)
    for i, (f, ei, fi, li, top) in enumerate(allVis):
      if f == fl and ei is not None:
        if top != eInT:
          skipped.append(eggName(fv, *eggs[ei]))
          continue
        tm = tmSinceLstVisTo(allVis, i, toTop)
        if opts.lastVisitIsMR and not tm[2]:
          continue
        vf.write(join(',', (vn, fn, eVc, fi2time(eggs[ei][1])) + tm) + '\n')
        updateInWinCounts(eVc, tm[1])
        checkSubstrate(eVc, top)
        ne += 1
    if ne == 0:
      continue
    eVc = eggVcontrol(False)
    for i in rng.permutation(len(allVis)):
      f, ei, fi, li, top = allVis[i]
      if top == eInT and f == fl and ei is None and \
          not offEggBeforeAfterVis(allVis, i, eggsS, tlen):
        tm = tmSinceLstVisTo(allVis, i, toTop)
        if opts.lastVisitIsMR and not tm[2]:
          continue
        if tm[0] is not None:
          vf.write(join(',', (vn, fn, eVc, fi2time(fi)) + tm) + '\n')
          updateInWinCounts(eVc, tm[1])
          checkSubstrate(eVc, top)
          nc += 1
          if nc == opts.numControls:
            break
  if skipped:
    print "\nskipped eggs (wrong side) for %s: %s" \
      %(VISIT_FILE, ", ".join(skipped))

# - - -
# footer

def writeAllStats(vf, s, h):
  if opts.writeVisits:
    if opts.lastVisit:
      writeAllLastVisitToStats(vf, s, h)
    elif opts.visitsTo:
      pass
    else:
      writeAllByEggStats(vf, s, h)

def writeAllByEggStats(vf, bes, h):
  vf.write('\n\n%s%s\n' %(h['vfeCs'], h['besCs']))
  for vfe, s in bes:
    vf.write(join(',', vfe) + ',%d,%d,%.2f,%.2f\n' %s)

def writeAllLastVisitToStats(vf, s, h):
  vf.write('\n%s\n' %h['inW'])
  for eVc, inWin in s[0].iteritems():
    v, nv = inWin
    vf.write(eVc + ',%d,%d,%d,%s\n'
      %(v, nv, v+nv, "{:.2%}".format(v/float(v+nv))))

# - - -

# analyze multiple files
def analyze():
  with open(VISIT_FILE if opts.writeVisits else os.devnull, 'w', 1) as vf:
    h = writeStatsHeader(vf)
    s = []   # collects from multiple files
    for i, fn in enumerate(opts.file.split(',')):
      if i: print
      a = analyzeFile(fn)
      writeStatsForFile(vf, a, s, h)
    writeAllStats(vf, s, h)
    if opts.writeVisits:
      print "\nwrote %s" %VISIT_FILE

# analyze single file
def analyzeFile(fn):
  isVnCn, preds, afn = VID_NUM_CLASS_NUM.match(fn), True, fn
  ncf = "no classification file"
  if isVnCn:
    vnCn = [int(n) for n in fn.split('-')]
    fv, afn = FlyVideo(vn=vnCn[0]), 'v%d' %vnCn[0]
    cn = vnCn[1] if len(vnCn) == 2 else \
      max(classify.classificationFile(fv, maxCn=True), 1)
    fn = fv.filename(" class-%d.data" %cn)
    if os.path.isfile(fn):
      afn += ' c%d' %cn
    else:
      preds = False
      if len(vnCn) == 2:
        error("%s has %s %d" %(afn, ncf, cn))
  print "=== analyzing %s ===\n" %afn
  if preds:
    checkIsfile(fn)
    d = unpickle(fn)
    fv = fv or FlyVideo(fn=d['videofn'])
    ep, allPreds = d['expPath'], d['allPreds']
    classify.printVideoModels(fv, ep)
  else:
    d = ep = None
    if not opts.useRegions:
      error(ncf + " (only allowed with --useReg)")

  tlen = fv.tlen()
  fr = classify.frameRange(tlen, d)
  if opts.useRegions:
    print '  using regions (top, middle, bottom) for "on"/"off" substrate'
    regs = fv.regions()
    if regs is None:
      error('no regions in "on substrate" MAT-file')
    isOn, prOn = regs != 2, None
  else:
    isOn, prOn = np.zeros((2, tlen), np.uint8), dict(allPreds)
    for key, pred in allPreds:
      isOn[key] = pred > .5
  print "  on substrate: %d" %np.count_nonzero(isOn)
  if opts.showCmd:
    cmd = d.get('cmd')
    print "  cc command: %s" \
      %("not available" if cmd is None else " ".join(cmd))

  if ep:
    checkTestSet(ep, fv, isOn, fr)

  # fix isOn; analysis should generally be after this fix
  if opts.useRegions:
    chKeys = set()
  else:
    _majorityFilterTest()
    isOn, chKeys = majorityFilter(isOn, verbose=True)

  eggs, isBe = loadEggFile(fv, isOn) if opts.eggFile else ([], None)

  vis = analyzeVisits(fv, isOn, eggs, isBe)
  if opts.numControls:
    vis.extend(notBeforeEggVisits(fv, isOn, eggs))
    vis = sorted(vis, key=operator.itemgetter(0))

  if opts.visualizeVisits:
    if opts.allVisitsRowM:
      visualizeAllVisits(fv, isOn, eggs)
    else:
      visualizeVisits(fv, vis, eggs)

  flagged = {}
  for f in [0, 1]:
    pio, pfi = isOn[f, fr[0]], None
    for fi in xrange(*fr):
      io = isOn[f, fi]
      if io != pio:
        if pfi is not None and fi-pfi <= opts.flagSeqLenTh:
          flagged.update(((f, fi1), fi-pfi) for fi1 in xrange(pfi, fi))
        pio, pfi = io, fi
  for f, fi in eggs:
    if not isOn[f, fi]:
      for fi1 in xrange(fi, -1, -1):
        if isOn[f, fi1]: break
        flagged[(f, fi1)] = -1

  showProblemImages(fv, isOn, prOn, flagged, chKeys, eggs, isBe, vis, fr)

  return dict(fv=fv, eggs=eggs, vis=vis,
    allVis=allVisits(fv, isOn, eggs) if \
      opts.writeVisits and opts.lastVisit else None)

# - - -

# show image sequence for the given fly and frame range (for debugging)
# sample call: showSequence(fv, isOn, 1, (53752, 53820))
def showSequence(fv, isOn, f, fiRng):
  fisH, nc, imgs, inSap = 72/2, 10, [], fv.inSapKeys()
  for fi in range(*fiRng):
    key = (f, fi)
    img = fv.getFlyImage(key, fisH)[0]
    labelImg(img, isOn[key])
    hdr = '%d ' %fi + ('T' if fv.inTop(*key) else 'B') + \
      (' in' if key in inSap else '')
    imgs.append((img, hdr))
  imgL = combineImgs(imgs, nc=nc)[0]
  cv2.imshow("sequence (fly %d)" %fv.flyNum(f), imgL)
  cv2.waitKey(0)

# - - -

parseOptions()
analyze()


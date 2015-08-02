#
# cuda-convnet automation
#
# 13 Jan 2014 by Ulrich Stern
#
# terminology
# * experiment: multiple trainings of net
# * execution:  single training, creates model
#

import argparse, os, shutil, time, subprocess
import re, scipy.stats as st, numpy as np, numpy.random as nr
import cv2

from util import *
from common import *

# - - -

# note: convention here: directories have trailing '/'
CC_DIR = ccDir(dropOut=False)
LAYERS_DIR = CC_DIR+"onsubs-layers/"
CC_DATA_DIR = DATA_DIR+"cc-exp2/V0-7/"
STD_LAYER_DEF, STD_LAYER_PARAMS = "layers.cfg", "layer-params.cfg"

# customize the following

# notes:
# * usable variables: $saveDir$, $featDir$
# *  TRAIN only: $expDir$

# e.g., ttRng(0,1) yields '1,2,5,6,9,10,...,33,34'
def ttRng(*folds):
  flds = set(folds)
  return join(",", (i+1 for i in range(36) if i%4 in flds))
ARCH = "3c2f-2"
LAYER_DEF = LAYERS_DIR+ARCH+".cfg"
LAYER_PARAMS = LAYERS_DIR+ARCH+"-params.cfg"
TRAIN_RNG_INIT = TRAIN_RNG = ttRng(0)
TEST_RNG_WHILE_TRAINING = TEST_RNG = ttRng(3)
TRAIN = [ [
  CONVNET,
  "--data-path="+CC_DATA_DIR,
  "--save-path=$expDir$",
  "--train-range="+TRAIN_RNG_INIT,
  "--test-range="+TEST_RNG_WHILE_TRAINING,
  "--layer-def=$expDir$"+STD_LAYER_DEF,
  "--layer-params=$expDir$"+STD_LAYER_PARAMS,
  "--data-provider=cifar-cropped", "--test-freq=201",
  "--postrotate-size=64", "--crop-border=4", "--rand-mirror=1",
  "--rand-norm=1", "--rand-rotate=4",
  "--epochs=400"
], [
  "epsW", "0.0001"
], [
  CONVNET,
  "-f", "$saveDir$",
  "--train-range="+TRAIN_RNG,
  "--epochs=600"
], [
  "epsW", "0.00001"
], [
  CONVNET,
  "-f", "$saveDir$",
  "--train-range="+TRAIN_RNG,
  "--epochs=800"
], [
  CONVNET,
  "-f", "$saveDir$",
  "--test-only=1", "--logreg-name=logprob",
  "--test-range="+TEST_RNG,
  "--multiview-test=1", "--multirotate-test=1", "--multinorm-test=1"
], [
  SHOWNET,
  "-f", "$saveDir$",
  "--write-features=probs",
  "--feature-path=$featDir$",
  "--test-range="+TEST_RNG
] ]

PREDICT = (
  # note: --test-range appeared not to be needed
  SHOWNET,
  "-f", "$saveDir$",
  "--write-features=probs",
  "--feature-path=$featDir$",
  "--multiview-test=1", "--mirror-test=0",
  "--multirotate-test=0", "--multinorm-test=1"
)

# - - -

DEBUG, VERBOSE = False, False   # not used yet

# shared
CONVNET_REGEX = re.compile(r'(?<=/)%s(?=__)' %CONVNET_PREFIX)
PRED_DIR_PREFIX, OLD_PRED_DIR_PREFIX = 'pred', 'feat'
PRED_DIR_PREFIXES = frozenset([PRED_DIR_PREFIX, OLD_PRED_DIR_PREFIX])
PRED_DIR_REGEX = re.compile(r'(?<=/)(%s)(?=__)' %'|'.join(PRED_DIR_PREFIXES))

# run experiment
CL_VAR = re.compile(r'\$(\w+)\$')
SAVE_DIR_EX = re.compile(r'^Saving checkpoints to (\S+)$', re.M)

PARAMS_SET = frozenset(['epsW'])

# analysis
N_MDLS_MA = (5, 10, 20, 30)
ERROR_IMG_DIR, PRED_SUBDIR_PREFIX = "err-imgs/", "p"
PRED_SUBDIR = re.compile(r'^%s(\d+)$' %PRED_SUBDIR_PREFIX)
ERR_RATES_FILE = "errRates.csv"

# - - -

# globals
errRates = []

# - - -

def options():
  eds = allExpDirs()
  p = argparse.ArgumentParser(description='cuda-convnet automation.')
  addGpuOption(p)
  p.add_argument('--sleep', dest='sleep', type=int, default=0, metavar='M',
    help='sleep for the given number of minutes first')

  numExecs = 30
  g = p.add_argument_group('run experiment')
  g.add_argument('--run', dest='train', action='store_true',
    help='train multiple models')
  g.add_argument('-n', dest='numExecs', type=int, default=numExecs, metavar='N',
    help='number of models to train (default: %(default)s)')

  expDir, bootstrapRepeats, pOnThreshold, nMdlsMaShowErrs, contextSize = \
    eds[-1] if eds else None, 500, .5, 20, 8
  g = p.add_argument_group('analyze experiment or run more predictions')
  meg = g.add_mutually_exclusive_group()
  meg.add_argument('--ana', dest='analyze', action='store_true',
    help='analyze the directory given with -d')
  meg.add_argument('--pred', dest='predict', action='store_true',
    help='run more predictions for the directory given with -d')
  g.add_argument('-d', dest='expDir', default=expDir, metavar='D',
    help='experiment directory (default: %(default)s);' +
      ' use, e.g., "2" for 2nd most recent directory;' +
      ' can be comma-separated list for --ana; list can include' +
      ' predictions subdirectories (e.g., "%s1")' %PRED_SUBDIR_PREFIX)
  g.add_argument('--dd', dest='dataDir', default=None, metavar='D',
    help='data directory to run predictions for (default: data directory ' +
      'that was used during training)')
  g.add_argument('--sn', dest='subdirNum', type=int, metavar='N',
    nargs='?', const=0, default=None,
    help='predictions subdirectory number (default for --ana if only ' +
      '--sn given: greatest existing number, default for --pred: ' +
      'next number)')
  g.add_argument('-b', dest='bootstrapRepeats', type=int,
    default=bootstrapRepeats, metavar='N',
    help='number of bootstrap repeats (default: %(default)s)')
  g.add_argument('--fix', dest='fixSeed', action='store_true',
    help='fix random seed for bootstrap')
  g.add_argument('--pOnTh', dest='pOnThreshold', type=float,
    default=pOnThreshold, metavar='P',
    help='P("on") threshold for predicting "on" (default: %(default)s)')
  g.add_argument('--allErrs', dest='allErrors', action='store_true',
    help='show errors for each model')
  g.add_argument('--nm', dest='nMdlsMaShowErrs', type=int,
    choices=N_MDLS_MA, default=nMdlsMaShowErrs, metavar='N',
    help='show errors for model averaging of N models (default: %(default)s)')
  g.add_argument('--imgs', dest='errorImages', action='store_true',
    help='show error images')
  g.add_argument('--vprc', dest='showPageNums', action='store_true',
    help='show video numbers and fly image page, row, and column numbers ' +
      'of the error images (see --imgs)')
  g.add_argument('--onProbs', dest='showOnProbs', action='store_true',
    help='show P("on") (calculated using all models) for each '
      'error image (see --imgs)')
  g.add_argument('--ctx', dest='contextSize', type=int, metavar='N',
    nargs='?', const=contextSize, default=0,
    help='show context (N before/after images) of the error images ' +
      '(default: %(const)s)')
  g.add_argument('-w', dest='writeErrors', action='store_true',
    help='write errors to %s and error images to %s (if --imgs given)'
      %(HC_ERROR_FILE, ERROR_IMG_DIR))
  g.add_argument('--wer', dest='writeErrRates', metavar='N',
    nargs='?', const=True, default=False,
    help='write error rates to %s; ' %ERR_RATES_FILE +
      'use the optional comma-separated experiment names as ' +
      'column headers')

  g = p.add_argument_group('other commands')
  meg = g.add_mutually_exclusive_group()
  meg.add_argument('--ld', dest='listTBDelDirs', action='store_true',
    help='list the to-be-deleted experiment directories (see --del)')
  meg.add_argument('--del', dest='delDirs', action='store_true',
    help='delete experiment directories with no subdirectory or ' +
      'with just one model subdirectory')
  return p.parse_args()

# - - -

# creates and returns experiment directory
def createExpDir():
  expDir = expPath(time2str(format='%Y-%m-%d__%H-%M-%S'))
  if not os.path.exists(expDir):
    os.makedirs(expDir)

  shutil.copy(__file__, expDir)   # copy self

  return expDir

# returns the "save directory"
def getSaveDir(expDir):
  # note: alternative: determine directory that was just added
  #  (may be better; 20 models with 800 epochs each give about 5MB log file)
  with open(expDir+EXP_LOG) as lf:
    dirs = re.findall(SAVE_DIR_EX, lf.read())
    assert len(dirs)
    return dirs[-1]

# - - -

# returns the given command with the "command line" variable names replaced
#  with their values
def replaceClVars(cmd, clVs):
  def clVarVal(mo):
    val = getattr(clVs, mo.group(1))
    assert val is not None
    return val
  return [re.sub(CL_VAR, clVarVal, s) for s in cmd]

# modify layer-params file according to cmd (e.g., ["epsW", "0.0001"])
def modifyParams(cmd, expDir):
  fn = expDir+STD_LAYER_PARAMS
  with open(fn) as f:
    c = f.read()
    for var, val in zip(*[iter(cmd)]*2):
      c = re.sub('^'+var+r'=\S+$', var+'='+val, c, flags=re.M)
  with open(fn, 'w') as f:
    f.write(c)

# train cuda-convnet once
def trainCcOnce(expDir, lf, i):
  for s, d in [(LAYER_DEF, STD_LAYER_DEF), (LAYER_PARAMS, STD_LAYER_PARAMS)]:
    shutil.copyfile(s, expDir+d)
  clVs = anonObj()
  clVs.expDir, clVs.saveDir = expDir, None
  for cmd in TRAIN:
    cmd0 = cmd[0]
    if cmd0 in CC_CMDS:
      execute(addGpu(replaceClVars(cmd, clVs), opts), CC_DIR, lf)
        # cuda-convnet seems to sometimes give return code 0 despite errors
      if clVs.saveDir is None:
        clVs.saveDir = getSaveDir(expDir)
        clVs.featDir = re.sub(CONVNET_REGEX, PRED_DIR_PREFIX, clVs.saveDir)
    elif cmd0 in PARAMS_SET:
      modifyParams(cmd, expDir)
    else:
      raise InternalError(cmd0)

def experiment(ed): return "experiment %s" %ed

# train cuda-convnet multiple times
def trainCc():
  expDir = createExpDir()
  print experiment(basename(expDir))
  with open(expDir+EXP_LOG, 'w', 1) as lf:
    rev, fn = svnRevision()
    lf.write('%s revision: %s\n\n' %(fn, rev))
    for i in range(opts.numExecs):
      print "  execution %d (of %d)" %(i+1, opts.numExecs)
      lf.write(('\n' if i > 0 else '') + "=== execution %d (of %d) ===\n" \
        %(i+1, opts.numExecs))
      trainCcOnce(expDir, lf, i)

# - - -

def deleteDirs(doDel):
  print ('cleaning up (deleting)' if doDel else 'to-be-deleted') + \
    ' "empty" experiment directories:'
  noDel, chkDirs = True, []
  for fn in allExpDirs():
    fp, dirs = EXP_DIR+fn, []
    for f in os.listdir(fp):
      sfp = fp + '/' + f
      assert os.path.exists(sfp)
      if os.path.isdir(sfp): dirs.append(f)
    nCnDirs = sum(d.startswith('ConvNet__') for d in dirs)
    if not dirs or len(dirs) == nCnDirs == 1:
      print "  %s" %fn
      if doDel:
        shutil.rmtree(fp)
      noDel = False
    elif nCnDirs < 5:
      chkDirs.append((fn, nCnDirs))
  if noDel:
    print "  nothing to delete"
  if chkDirs:
    ml = max(len(t[0]) for t in chkDirs)
    print '\n"strange" experiment directories (with numbers of ' + \
      'ConvNet subdirs):\n%s' %('\n'.join(
      ('  %-'+str(ml)+'s  (%d)') %t for t in chkDirs))

# - - -

def requireNotNone(var=None, toShow=None):
  if var is None:
    raise InternalError("required info%s is not available"
      %(" to show %s" %toShow if toShow else ""))

def vnKeyPrc(db, bNums, b, idx, require=False):
  bNum = bNums[b]
  vnKey, prc = db.vnKey(bNum, idx), db.pageRowColumn(bNum, idx)
  if require and not (vnKey and prc):
    requireNotNone()
  return vnKey, prc, bNum

def showErrorImages(db, bNums, errSet, pOnErrs):
  cSz, cpr, fis = opts.contextSize, 1, db.newImgSize()
  nr, nc = 10, cpr*(2*cSz+1) if cSz else 10
  n = nr*cpr if cSz else nr*nc
  hces = readHcErrors()[0]

  def addImgs(imgs, vnKey, num):
    vn, (f, fi) = vnKey
    fv = FlyVideo.getInstance(vn)
    for i in range(abs(num)):
      fi1 = fi+num+i if num < 0 else fi+1+i
      if fi1 >= 0 and fi1 < fv.tlen():
        dfi = fi1-fi
        imgs.append((fv.getFlyImage((f, fi1), fis/2)[0],
          '%d' %(fi1 if dfi == num and num < 0 else dfi)))
  def showImages(errs):
    imgs, hdrFn, hdrShw = [], None, None
    for b, idx in errs:
      vnKey, prc, bNum = vnKeyPrc(db, bNums, b, idx)
      if cSz:
        requireNotNone(vnKey, "context")
        addImgs(imgs, vnKey, -cSz)
      img = db.getImage(bNum, idx, label=True)
      if vnKey and prc and (vnKey[0], prc) in hces:
        putText(img, 'x', tupleSub(bottomRight(img), (2,3)), (-1, 0),
          TXT_STL_TAG)
      if opts.showPageNums:
        requireNotNone(prc, "page numbers")
        hdr = ('v%d ' %vnKey[0] if vnKey else '') + '%dr%dc%d'%prc
      else:
        hdr = 'b%d %d' %(bNum, idx)
      hdrFn, hdrShw = hdrFn or hdr, hdrShw or 'b%d %d' %(bNum, idx)
      if opts.showOnProbs:
        pOn = pOnErrs.get((b, idx))
        hdr = '{:.0%}'.format(pOn) if pOn else '-'
      imgs.append((img, hdr))
      if cSz:
        addImgs(imgs, vnKey, cSz)
    imgL = combineImgs(imgs, nc)[0]
    cv2.imshow('errors starting with "%s"' %hdrShw, imgL)
    if opts.writeErrors:
      cv2.imwrite(ERROR_IMG_DIR+"errors %s.png" %hdrFn, imgL)
 
  errs = sorted(errSet)
  if opts.showPageNums:
    errs = [e[1] for e in sorted(zip(vnPrcList(db, bNums, errs), errs))]
  [showImages(errs[i:i+n]) for i in range(0, len(errs), n)]
  cv2.waitKey(0)

# yields experiment directory, path, name (default: directory), and predictions
#  subdir number (default: None)
def expDirs():
  eds, led, aeds = opts.expDir.split(','), None, allExpDirs()
  if isinstance(opts.writeErrRates, str):
    ens = opts.writeErrRates.split(',')
    if len(ens) != len(eds):
      error('numbers of experiments and experiment names must match')
    edns = zip(eds, ens)
  else:
    edns = zip(eds, eds)
  for ed, en in edns:
    mo, sn = PRED_SUBDIR.match(ed), None
    if mo:
      if led is None:
        error('experiment directory must be given before predictions subdir')
      ed, sn = led, int(mo.group(1))
    else:
      ed = led = expandExpDir(ed, aeds)
    yield ed, expPath(ed), en, sn

# returns predictions dirs and the greatest predictions subdir number (0: none)
def predDirs(ep):
  pds = [d for d in sorted(os.listdir(ep)) if \
    os.path.isdir(ep+d) and \
    any(d.startswith(p) for p in PRED_DIR_PREFIXES)]
  if pds:
    sns = [int(sn) for sn in multiMatch(PRED_SUBDIR, os.listdir(ep+d))]
  return pds, max(sns) if pds and sns else 0

# returns predictions subdir
def predSubdir(sn): return "%s%d" %(PRED_SUBDIR_PREFIX, sn)
# returns predictions log fn
def predLogfn(sn): return "__%s.log" %predSubdir(sn)

def readModels(ep, sn, dtDir, dtDirP, ccos, bSz):
  bNums = None
  lblsByB, predsByMB, errsByMB, nErrsByM = [], [], [], []
    # ByB: by batch (index), ByMB: by model and batch, ByM: by model
  pdirs, maxSn = predDirs(ep)
  sd = predSubdir(sn or maxSn)
  for pdir in pdirs:
    pp = ep+pdir+'/' + ('' if sn is None else sd+'/')
    if not os.path.isdir(pp):
      error("predictions subdir does not exist%s" %(
        " (max. subdir number: %d)" %maxSn if maxSn > 0 else ""))
    if bNums is None:
      bNums = DataBatch.batchNumsForDir(pp)
      nb = len(bNums)
    else:
      assert bNums == DataBatch.batchNumsForDir(pp)
    predsM, errsM, nErrsM  = nb*[None], nb*[None], 0
    for i, bNum in enumerate(bNums):
      lbls, preds = readPredictionBatch(pp, bNum)
      if not lblsByB:
        dtMult = lbls.size / bSz
        assert lbls.size % bSz == 0
        print "  data directory: %s     initial epochs: %s     gpu: %s" \
          %(basename(dtDir), ccos['epochs'], ccos.get('gpu', 'default'))
        if sn is not None:
          print "    predictions subdir: %s%s" %(sd, "" if not dtDirP else
            "     (separate) data directory: %s" %basename(dtDirP))
        print "  batch numbers:  (batch size: %d)" %bSz
        print "    initial training: %s" %ccos['train-range']
        print "    test:             %s" %join(",", bNums)
        def joinCcos(*os):
          os = [o.partition('=') for o in os]
          return ", ".join("%s: %s" %(o[0], ccos.get(o[0], o[2])) for o in os
            if ccos.get(o[0], o[2]))
        print "  " + joinCcos('postrotate-size=default', 'crop-border')
        print "    " + joinCcos('rand-norm', 'rand-rotate', 'rand-mirror=[1]')
        print "\nnumber of models: %d" %len(pdirs)
        print "number of test images: %d, data multiplier: %d" %(bSz*nb, dtMult)
      assert lbls.size == bSz * dtMult
      lbls, preds = multiviewAverage(lbls, preds, dtMult)
      if len(lblsByB) < nb:
        lblsByB.append(lbls)
      else:
        assert np.array_equal(lbls, lblsByB[i])
      predsM[i] = preds
      errs = (preds.argmax(axis=1) != lbls).nonzero()[0]
      errsM[i] = errs
      nErrsM += len(errs)
    predsByMB.append(predsM)
    errsByMB.append(errsM)
    nErrsByM.append(nErrsM)
  return bNums, lblsByB, predsByMB, errsByMB, nErrsByM

def numLbls(lblsByB): return len(lblsByB) * lblsByB[0].size

def reportErr(msg, es, pcnt=False, sem=True):
  mean = len(es) > 1
  f = "{:.3%}" if pcnt else "{:.2f}"
  if mean:
    se = ", SEM " + f.format(st.sem(es)) if sem else \
         ", SEB " + f.format(np.std(es, ddof=1))
  print ("mean " if mean else "") + msg + ": " + \
    f.format(np.mean(es)) + (se if mean else "")

# report "number of errors" and "error rate" given list with numbers of
#  errors; for more than one number, mean is reported
def reportErrs(nes, nLbls, label, sem=True):
  reportErr("number of errors", nes, sem=sem)
  ers = np.array(nes)/float(nLbls)
  reportErr("error rate", ers, pcnt=True, sem=sem)
  errRates.append([label] + list(ers))

# report precision and recall
def reportPrecRec(precs, recs):
  reportErr("precision", precs, pcnt=True, sem=False)
  reportErr("recall", recs, pcnt=True, sem=False)

def bootstrapRepeats():
  return "bootstrap repeats: %d" %opts.bootstrapRepeats

# notes:
# * no randomization for opts.bootstrapRepeats=1
# * see An Introduction to Statistical Learning for precision and recall
def modelAverage(rng, nMdlsMA, lblsByB, predsByMB, en, errSet=False):
  nRepeats = opts.bootstrapRepeats
  lbls, preds = np.array(lblsByB), np.array(predsByMB)
  nes, es = nRepeats*[None], set()
  nP, precs, recs = np.count_nonzero(lbls), [], []
  for i in range(nRepeats):
    mdls = np.arange(nMdlsMA) if nRepeats == 1 else \
      rng.randint(len(predsByMB), size=nMdlsMA)
    on = preds[mdls].sum(axis=0)[:,:,1] > opts.pOnThreshold*nMdlsMA
    errs = (on != lbls).nonzero()
    if DEBUG:
      errs1 = (preds[mdls].sum(axis=0).argmax(axis=2) != lbls).nonzero()
      assert zip(*errs) == zip(*errs1)
        # note: likely fails for pOnThreshold != .5
    nes[i] = len(errs[0])
    if errSet:
      es.update(zip(*errs))
    nTP = float(np.count_nonzero(np.logical_and(on == lbls, on == 1)))
    nPS = np.count_nonzero(on)
    assert nes[i] == nP+nPS-2*nTP
    precs.append(nTP/nPS)
    recs.append(nTP/nP)
  print "model averaging of %d models:  (%s)" %(nMdlsMA, bootstrapRepeats())
  reportErrs(nes, numLbls(lblsByB), '%s-ma%d' %(en, nMdlsMA), sem=False)
  reportPrecRec(precs, recs)
  return es

# converts (b, idx) collection to (vn, (p, r, c)) list
def vnPrcList(db, bNums, keys):
  vnPrc = []
  for b, idx in keys:
    vnKey, prc = vnKeyPrc(db, bNums, b, idx, require=True)[:2]
    vnPrc.append((vnKey[0], prc))
  return vnPrc

def writeCommand(f):
  f.write('# command: %s\n' %' '.join(sys.argv))

def writeErrRates():
  print "\nwriting error rates to %s" %ERR_RATES_FILE
  with open(ERR_RATES_FILE, 'w') as f:
    writeCommand(f)
    f.write('# experiments: %s\n' %', '.join(
      (ed if en == ed else '%s=%s' %(en, ed +
        ('' if sn is None else ' %s' %predSubdir(sn))))
      for ed, ep, en, sn in expDirs()))
    ml = max(len(l) for l in errRates)
    ers = [[rw[0]] + ['%.4f'%(e*100) for e in rw[1:]] + [''] * (ml-len(rw))
        for rw in errRates]
    # group by number of models for model averaging
    idxs = np.arange(len(ers)).reshape(-1, 1+len(N_MDLS_MA)).T.flatten()
    ers1 = [ers[i] for i in idxs]
    for rw in zip(*ers1):
      f.write(','.join(rw) + '\n')

def analyze():
  for i, (ed, ep, en, sn) in enumerate(expDirs()):
    if i > 0: print
    analyzeExp(ed, ep, en, sn)
  if opts.writeErrRates:
    writeErrRates()

def analyzeExp(ed, ep, en, sn):
  print "=== analyzing %s ===\n" %experiment(ed)
  sn = sn or opts.subdirNum
  ccos, ccosP = ccOpts(ep), ccOpts(ep, predLogfn(sn)) if sn else {}
  dtDir, dtDirP = ccos['data-path'], ccosP.get('data-path')
  db = DataBatch(dtDirP or dtDir, labelSet=LABEL_SET)
  bSz = db.batchSize()
  bNums, lblsByB, predsByMB, errsByMB, nErrsByM = \
    readModels(ep, sn, dtDir, dtDirP, ccos, bSz)
 
  if not lblsByB:
    print "no models found"
    return

  nLbls, nb, nMdls = numLbls(lblsByB), len(bNums), len(predsByMB)
  errSet = set()   # elements: (batch index, image index)
  if opts.allErrors:
    print "\nindexes of errors for each model:"
    for i, bNum in enumerate(bNums):
      print "batch %d:" %(bNum)
      for errsM in errsByMB:
        es = errsM[i]
        print "  %s" %es
        errSet |= set(zip(len(es)*[i], es))

  print
  reportErrs(nErrsByM, nLbls, en)

  print "\n---"
  rng = nr.RandomState(42 if opts.fixSeed else None)
  if opts.fixSeed:
    print "note: using fixed random seed for bootstrap"

  eMaByB, wErrSet = [[] for i in range(nb)], None
  for nMdlsMA in N_MDLS_MA:
    if nMdlsMA > nMdls:
      break
    es = modelAverage(rng, nMdlsMA, lblsByB, predsByMB, en,
      errSet=nMdlsMA==opts.nMdlsMaShowErrs)
    if es:
      for b, idx in sorted(es):
        eMaByB[b].append(idx)
      errSet |= es
      wErrSet = es
    print

  print ('indexes of errors with model averaging of %d models and "on" ' +
    'probabilities:') %opts.nMdlsMaShowErrs
  pOnErrs = {}
  if nMdls < opts.nMdlsMaShowErrs:
    print "not calculated (only %s so far)" %nItems(nMdls, "model")
  else:
    print '(%s; "on" probabilities calculated using all models)' \
      %(bootstrapRepeats())
    onLbl = db.getLabels().index(ON_LABEL)
    predsByB = np.array(predsByMB).sum(axis=0)
    for i, bNum in enumerate(bNums):
      pOn = predsByB[i,:,onLbl] / nMdls
      print "  %d: [%s]" %(bNum, ", ".join(
        '{:d}: {:.0%}'.format(idx, float(pOn[idx])) for idx in eMaByB[i]))
      pOnErrs.update(((i, idx), pOn[idx]) for idx in eMaByB[i])
    print "total number of errors: %d" %(sum(len(es) for es in eMaByB))

  if opts.writeErrors:
    print "\nwriting errors to %s" %(HC_ERROR_FILE)
    with open(HC_ERROR_FILE, 'w') as f:
      f.write('# video,page,row,column\n')
      writeCommand(f)
      f.write('# experiment: %s\n' %ed)
      for vn, prc in sorted(vnPrcList(db, bNums, wErrSet)):
        f.write(join(",", [vn] + list(prc)) + "\n")

  if opts.errorImages:
    showErrorImages(db, bNums, errSet, pOnErrs)

# - - -

def predict():
  ed, ep = expDirs().next()[:2]
  print "running predictions for %s" %experiment(ed)
  pdirs, maxSn = predDirs(ep)
  sn = opts.subdirNum if opts.subdirNum else maxSn + 1
  sd = predSubdir(sn)
  print "  predictions subdir: %s\n" %sd
  with open(ep + predLogfn(sn), 'w', 1) as lf:
    for pdir in pdirs:
      print "  %s" %pdir
      clVs, pp = anonObj(), ep+pdir
      clVs.featDir = pp + '/%s/'%sd
      clVs.saveDir = re.sub(PRED_DIR_REGEX, CONVNET_PREFIX, pp)
      cmd = replaceClVars(PREDICT, clVs)
      if opts.dataDir:
        nb = len(DataBatch.batchNumsForDir(pp))
        cmd += ["--data-path="+opts.dataDir, "--test-range=1-%d" %nb]
      execute(addGpu(cmd, opts), CC_DIR, lf)

# - - -

def sleep():
  sl = opts.sleep
  if sl:
    print "sleeping for %s..." %nItems(sl, "minute")
    time.sleep(sl * 60)
    print

# - - -

def testModifyParams():
  expDir = expPath(allExpDirs()[-1])
  modifyParams(["epsW", "0.0001"], expDir)

def testSTD():
  def std(a):
    return np.sqrt(((a-np.mean(a))**2).sum() / float(a.size-1))
  for i in range(5):
    a = nr.randint(10, size=10)
    print "%s: np.std=%.4f std=%.4f" %(a, np.std(a, ddof=1), std(a))

if False:
  testModifyParams()
  testSTD()
  sys.exit()

# - - -

opts = options()
sleep()
if opts.delDirs or opts.listTBDelDirs:
  deleteDirs(opts.delDirs)
elif opts.analyze:
  analyze()
elif opts.predict:
  predict()
elif opts.train:
  trainCc()
else:
  print "nothing to do (call with -h for help)"


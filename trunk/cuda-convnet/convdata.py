# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from data import *
import numpy.random as nr, numpy as n, random as r
import sys
import cv2

VERBOSE = True
DEBUG = False

# - - -

def _img_size_num_cols(dp=None, n_vis=None):
    if n_vis is None:
        n_vis = dp.batch_meta['num_vis']
    sr = int(n.sqrt(n_vis))
    if n_vis == sr**2:
        return sr, 1
    else:
        assert n_vis % 3 == 0
        n_vis = n_vis/3
        sr = int(n.sqrt(n_vis))
        assert n_vis == sr**2
        return sr, 3

def _rotate_img(img, angle):
    cntr = tuple(n.array(img.shape[:2]) / 2)
    mat = cv2.getRotationMatrix2D(cntr, angle, 1.)
    return cv2.warpAffine(img, mat, img.shape[:2], flags=cv2.INTER_LINEAR)

# note: could be moved into, e.g., CroppedCIFARDataProvider
def _show_imgs(dp, data):
    nver = 4   # number of versions of original image to show
    sampleForOrig = True   # sample vs. start of batch; default: True
    rws, cls, d, nimgs = 8, 12, 10, data.shape[1]
    assert rws*cls % nver == 0
    isz, nc = _img_size_num_cols(n_vis=data.shape[0])
    imgL = n.zeros((rws*(isz+d), cls*(isz+d), nc), n.single)
    dm, hl = dp.get_data_mult(), None
    assert nimgs % dm == 0
    def sample_vs_range(np, ns):
        return r.sample(xrange(np), ns) if sampleForOrig else xrange(ns)
    if dm >= nver:
        nOrigImgs, nOrigSmpl = nimgs / dm, rws*cls / nver
        idxs, origIdxs = [], sample_vs_range(nOrigImgs, nOrigSmpl)
        for oi in origIdxs:
            idxs.extend(n.array([0] + r.sample(xrange(1,dm), nver-1))*nOrigImgs + oi)
        hl = nOrigSmpl * ([True] + (nver-1)*[False])
    else:
        idxs = sample_vs_range(nimgs, rws*cls)
    data = dp.get_plottable_data(data[:, idxs])
    print "indexes of images shown: %s" %idxs
    sys.stdout.flush()
    for rw in range(rws):
        for cl in range(cls):
            x, y = d/2+cl*(isz+d), d/2+rw*(isz+d)
            x2, y2 = x+isz, y+isz
            imgL[y:y2, x:x2, :] = data[cl+rw*cls]
            if hl and hl[cl+rw*cls]:
                cv2.rectangle(imgL, (x, y), (x2, y2), 255, 1)
    imname = "next batch: %d samples out of %d%s" %(rws*cls, nimgs,
        " (original images highlighted)" if hl else "")
    cv2.imshow(imname, imgL)

# - - -

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.img_size, self.num_colors = _img_size_num_cols(self)
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, numColors) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], self.num_colors, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

# - - -

class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        if VERBOSE:
            print "%s ctor" %self.__class__.__name__
            print "batch_range: %s" %self.batch_range
        self.img_size, self.num_colors = _img_size_num_cols(self)
        self.border_size, self.rand_mirror, self.rand_rotate = (dp_params[n] for n in
            ('crop_border', 'rand_mirror', 'rand_rotate'))
        if self.rand_rotate:
            self.img_size_after_rotate = (int(self.img_size / n.sqrt(2)) | 1) - 1
            # e.g., img_size 92 -> img_size_after_rotate 64
        if dp_params['postrotate_size']:
            self.img_size_after_rotate = dp_params['postrotate_size']
        self.rotate_border_size = (self.img_size - self.img_size_after_rotate)/2 if self.rand_rotate else 0
        self.inner_size = self.img_size - (self.rotate_border_size + self.border_size)*2

        self.multiview, self.multinorm, self.multirotate, self.mirror_test = (dp_params[n] and test
            for n in ('multiview_test', 'multinorm_test', 'multirotate_test', 'mirror_test'))
        self.num_views, self.num_norms = 5*(2 if self.mirror_test else 1), 4
        self.num_rotates = min(16, self.rand_rotate) if self.rand_rotate > 1 else 16
        self.data_mult_norm = self.num_norms if self.multinorm else 1
        self.data_mult_rotate = self.num_rotates if self.multirotate else 1
        self.data_mult = (self.num_views if self.multiview else 1) * self.data_mult_norm * self.data_mult_rotate
        if VERBOSE:
            print "data_mult: %d, num_rotates: %d, rand_rotate: %d, rand_mirror: %d, border_size: %d" \
                %(self.data_mult, self.num_rotates, self.rand_rotate, self.rand_mirror, self.border_size)

        self.rand_norm = dp_params['rand_norm']
        self.min_max, min_max_all, min_dist = [self.batch_meta.get(k) for k in \
            ['YL_min_max', 'YL_min_max_all', 'YL_min_distance_black_white']]
        if self.rand_norm:
            self.min_norm, self.max_norm = min_max_all[0] - min_dist, min_max_all[1] + min_dist
            if VERBOSE:
                print "min_norm: %d, max_norm: %d" %(self.min_norm, self.max_norm)

        for d in self.data_dic:
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]
        if VERBOSE:
            print "test: %s, cropped_data: %s" %(test, self.cropped_data[0].shape)
        self.norm_data = n.zeros((self.img_size_after_rotate**2, self.data_dic[0]['data'].shape[1]*self.data_mult_rotate*self.data_mult_norm), dtype=n.single)
        self.rotated_data = [n.zeros((self.img_size_after_rotate**2, self.data_dic[0]['data'].shape[1]*self.data_mult_rotate), dtype=n.single) for x in range(2)]

        self.batches_generated = 0
        bSz = self.rotate_border_size + self.border_size
        self.data_mean = self.batch_meta['data_mean'].reshape((self.num_colors,self.img_size,self.img_size))[:,bSz:bSz+self.inner_size,bSz:bSz+self.inner_size].reshape((self.get_data_dims(), 1))
          # note: center patch mean used for all patches

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        rotated = self.rotated_data[self.batches_generated % 2]
          # note: not sure it matters to have 2 ndarrays for the rotated data
        cropped = self.cropped_data[self.batches_generated % 2]

        data = datadic['data']
        if self.rand_rotate:
            data = self.__rand_rotate(data, rotated)
        if self.rand_norm:
            data = self.__rand_norm(data, self.norm_data, batchnum)
        self.__trim_borders(data, cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        if DEBUG:
            _show_imgs(self, cropped)
            cv2.waitKey(0)
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * self.num_colors if idx == 0 else 1

    def get_data_mult(self):
        return self.data_mult

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, numColors) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], self.num_colors, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

    def __trim_borders(self, x, target):
        imSz = n.sqrt(x.shape[0] / self.num_colors)
        y = x.reshape(self.num_colors, imSz, imSz, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                bSz, off = self.border_size, self.border_size/2
                start_positions = [(bSz, bSz), (bSz-off, bSz-off), (bSz-off, bSz+off), (bSz+off, bSz-off), (bSz+off, bSz+off)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/(2 if self.mirror_test else 1)):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    if self.mirror_test:
                        target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]):   # for each case (image)
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0 and self.rand_mirror: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))

    # random normalization
    def __rand_norm(self, data, target, batchnum):
        assert target.shape[1] == data.shape[1]*self.data_mult_norm
        mi_ma = self.min_max[batchnum-1]
        if self.test:
            if self.multinorm:
                mi_ma = n.array(mi_ma).transpose()
                dlts = mi_ma[1] - mi_ma[0]
                nmm = len(dlts)
                min_r, max_r = (n.repeat(v, nmm) for v in (self.min_norm, self.max_norm))
                  # sample values: self.min_norm: 0, self.max_norm: 196
                # normalizations: original, darker, brighter, "max" contrast
                nrm_orig, nrm_dark, nrm_bright, nrm_max_contr = \
                    (None, None), (min_r, min_r + dlts), (max_r - dlts, max_r), (min_r, max_r)
                new_mi_ma = [nrm_orig, nrm_dark, nrm_bright, nrm_max_contr]
                #new_mi_ma = 2*[nrm_orig] + 2*[nrm_max_contr]
                if DEBUG:
                    print "batch %d" %batchnum
                    for c in range(10):
                        img = data[:,c]
                        cmi, cma, (mi, ma) = n.amin(img), n.amax(img), mi_ma[:,c]
                        print "  img %d: %d %d, calc: %d %d" %(c, mi, ma, cmi, cma)
                for i, new_mm in enumerate(new_mi_ma):
                    for c in xrange(data.shape[1]):
                        if new_mm[0] is None:
                            nimg = data[:,c]
                        else:
                            new_mi, new_ma = [e[c%nmm] for e in new_mm]
                            nimg = n.squeeze(cv2.normalize(data[:,c], None, new_mi, new_ma, cv2.NORM_MINMAX, -1))
                        target[:, c+i*data.shape[1]] = nimg
            else:
                target[:,:] = data
        else:
            for c in xrange(data.shape[1]):   # for each case (image)
                mi, ma = mi_ma[c]
                dlt = nr.random_integers(ma-mi, self.max_norm-self.min_norm)
                new_mi = nr.random_integers(self.min_norm, self.max_norm-dlt)
                if DEBUG and c == 0:
                    img = data[:,c]
                    pmi, pma, ps = n.amin(img), n.amax(img), img.shape
                target[:,c] = n.squeeze(cv2.normalize(data[:,c], None, new_mi, new_mi+dlt, cv2.NORM_MINMAX, -1))
                if DEBUG and c == 0:
                    img = target[:,c]
                    print "batch %d, img %d: %d %d (calc: %d %d, shp: %s) -> %d %d (calc: %d %d, shp: %s)" \
                        %(batchnum, c, mi, ma, pmi, pma, ps, new_mi, new_mi+dlt, n.amin(img), n.amax(img), img.shape)
        return target

    # random rotation
    def __rand_rotate(self, data, target):
        assert self.num_colors == 1
        assert target.shape[1] == data.shape[1]*self.data_mult_rotate
        d = data.reshape(self.img_size, self.img_size, data.shape[1])
        isar, rbs = self.img_size_after_rotate, self.rotate_border_size
        if self.test:
            if self.multirotate:
                for i in range(self.num_rotates):
                    for c in xrange(data.shape[1]):
                        rimg = _rotate_img(d[:,:,c], i*360/self.num_rotates)[rbs:rbs+isar, rbs:rbs+isar]
                        target[:, c+i*data.shape[1]] = rimg.reshape((isar**2,))
            else:
                target[:,:] = d[rbs:rbs+isar, rbs:rbs+isar, :].reshape((isar**2, -1))
        else:
            for c in xrange(data.shape[1]):   # for each case (image)
                rimg = _rotate_img(d[:,:,c], nr.randint(self.rand_rotate)*360/self.rand_rotate)[rbs:rbs+isar, rbs:rbs+isar]
                target[:,c] = rimg.reshape((isar**2,))
        return target

# - - -
 
class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1

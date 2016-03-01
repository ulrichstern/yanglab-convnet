This page does not give detailed instruction for the install; instead, it gives an overview of the project and explains several aspects of the data and code in the hope that this will be sufficient for researchers and programmers to get things working.  A link to a paper about this project will be made available.  In a nutshell, our code allows you to classify each frame of a video using convolutional neural networks (CNNs).  The CNNs are implemented by [cuda-convnet](http://code.google.com/p/cuda-convnet/).  Our code executes cuda-convnet scripts (e.g., to train CNNs) and generates data files (with images) that cuda-convnet can read (both human-labeled for training and non-labeled for having cuda-convnet predict).

## yanglab-convnet directories ##
If you check out yanglab-convnet, you get the following directories
  * cuda-convnet
    * contains all the files in cuda-convnet that we modified
    * after you got cuda-convnet working (Alex has good documentation for this), you should replace the cuda-convnet files with the files in this directory
  * onsubs
    * our scripts, see Overview of code below for details
    * you have to customize the paths used by the scripts for your situation
  * onsubs-data
    * data we used in the paper -- images labeled with whether flies (_Drosophila_, the "system" we study in our lab) are "on" or "off" egg-laying substrates
    * V0-7: 36 batches with 600 images each
      * used for training and validation
    * V0-7\_test: 9 batches with 600 images each
      * used for test
      * note: the test batches are 1-9; batches 10-36 in this directory are just copied from V0-7 to keep the layout identical
  * onsubs-layers
    * our cuda-convnet layer definition and layer parameter files
      * the former defines the CNN architecture, the latter the learning parameters
    * used for the paper: 3c2f-2.cfg (3 conv, 2 fc layers) and 3c2f-2-params.cfg

## Overview of code ##

Note: each script below can be called with "-h" to see all command-line options.

### onSubstrate.py ###
  * we used this script to generate the labeled data (provided in onsubs-data directory) from our fly videos
    * it has a very simple UI to allow human labeling
  * you need to run this script only if you want to generate labeled data yourself
    * since the fly videos are large (~ 1GB each), they would have gotten use over Google's quota, and we did not include them in yanglab-convnet

### autoCc.py ###
  * automates training of multiple nets
    * autoCc stands for "automate cuda-convnet," so this script calls cuda-convnet (both convnet.py and shownet.py)
    * each training results in an _experiment_ directory with
      * all the nets, prediction files for each net
      * a log file for the experiment with all cuda-convnet output
      * the cuda-convnet layer definition and layer parameter files used
        * the parameter file is edited by autoCc.py to, e.g., change learning rate after a certain number of epochs
    * the code in the beginning of the file can be edited to change the training
  * you can run additional predictions for previously trained nets
    * we used this, e.g., to figure out what data augmentation for test worked best
  * measures error rates of multiple nets
    * has model averaging (including the bootstrap) and data augmentation for test
    * error rates are calculated based on the prediction files

### classify.py ###
  * classifies "on"/"off" substrate for each fly and frame in a fly video
    * calls cuda-convnet's shownet.py to generate predictions
    * uses positional information from [Ctrax](http://ctrax.sourceforge.net/) (the tracking software we use) to crop small fly images (with fly in center) from the full frame

### analyze.py ###
  * fly behavior analysis based on Ctrax trajectories and classify.py's "on"/"off" classification
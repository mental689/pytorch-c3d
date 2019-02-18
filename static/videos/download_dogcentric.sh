#!/usr/bin/env bash

wget --recursive --no-parent -R "index.html*" ftp://robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/

PARENT=`pwd`
## Under dogcentric directory, there are 4 sub-directories corresponding to 4 subjects (dogs).
## The sub-directories seem to be named by dog names, and the segmented videos are archived in tar balls.
## The following code do the following:
## Go to each dog name directory and then extract segmented videos from tar balls.
for name in 'Hime' 'Ku' 'Ringo' 'Ryu' ; do
    cd ${PARENT}/robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/${name}/
    for i in `find ./*.tar.gz`; do
        tar zxvf ${i}
    done
done

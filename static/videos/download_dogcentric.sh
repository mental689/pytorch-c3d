#!/usr/bin/env bash

wget --recursive --no-parent -R "index.html*" ftp://robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/

PARENT=`pwd`
for name in 'Hime' 'Ku' 'Ringo' 'Ryu' ; do
    cd ${PARENT}/robotics-ftp.ait.kyushu-u.ac.jp/dogcentric/${name}/
    for i in `find ./*.tar.gz`; do
        tar zxvf ${i}
    done
done

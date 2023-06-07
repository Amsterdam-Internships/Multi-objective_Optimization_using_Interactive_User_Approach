#!/bin/bash
#

docker run --rm -ti \
        -v `pwd`/:/src \
	    -v /data/$HOSTNAME/luiraf/:/data/ \
        luisa/guts $@

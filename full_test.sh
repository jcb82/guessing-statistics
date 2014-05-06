#!/bin/sh

sort_disk='-T ~/temp/'

sort ${sort_disk} |\
uniq -c |\
sort ${sort_disk} -nr |\
sed  -r 's/^[ ]+//' |\
sed  -r 's/ /,/'

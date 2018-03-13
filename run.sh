#!/bin/sh
xvfb-run -a -s "-screen 0 1400x900x24 +extension RANDR" -- $*

# ps -ef | grep
# setsid
# nohup

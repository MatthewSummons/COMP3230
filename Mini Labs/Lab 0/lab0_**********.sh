#!/bin/zsh

FILENAME=auth.log

if [ ! -f ./$FILENAME ]
then
	echo The file $FILENAME does not exist
else
	# Find all failed login attempts, record the IPs of the offenders and save into a file
	grep "Failed" $FILENAME | grep -oE "[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" - > suspicious_ips.txt
fi

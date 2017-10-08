# Copyright (c) 2015 Angus H. (4148)
# Distributed under the GNU General Public License v3.0 (GPLv3).

from datetime import date, timedelta
from random import randint
from time import sleep
import sys
import subprocess
import os

# returns a date string for the date that is N days before STARTDATE
def get_date_string(n, startdate):
	d = startdate - timedelta(days=n)
	rtn = d.strftime("%a %b %d %X %Y %z -0400")
	return rtn

# main app
def main(argv):
	print 'Starting ...'
	if len(argv) < 1 or len(argv) > 2:
		print "Error: Bad input."
		sys.exit(1)
	n = int(argv[0])
	if len(argv) == 1:
		startdate = date.today()
	if len(argv) == 2:
		startdate = date(int(argv[1][0:4]), int(argv[1][5:7]), int(argv[1][8:10]))
	i = 0
	print 'Start Date Calculated', startdate
	while i <= n:
		curdate = get_date_string(i, startdate)
		print 'Current Date Calculated', curdate
		num_commits = randint(1, 10)
		print 'Number of Commits', num_commits
		for commit in range(0, num_commits):
			subprocess.call("echo '" + curdate + str(randint(0, 1000000)) +"' > realwork.txt; echo 'SS'; git add realwork.txt; GIT_AUTHOR_DATE='" + curdate + "' GIT_COMMITTER_DATE='" + curdate + "' git commit -m 'update'; git push;", shell=True)
			print 'Cuurent Commit Number', commit
			sleep(.5)
		i += 1
	print 'All commits done'
	subprocess.call("git rm realwork.txt; git commit -m 'delete'; git push;", shell=True)

if __name__ == "__main__":
	main(sys.argv[1:])

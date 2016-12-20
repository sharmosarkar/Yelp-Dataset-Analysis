#!/bin/bash
# DEBUG, in case \r error occurs
#awk '{ sub("\r$", ""); print }' git-copy.sh > git-copy2.sh
#mv git-copy2.sh git-copy.sh
echo 'Starting Script'
echo $1
echo $2
rem=$(( $2 % 2 ))
bakup=".bak"
FILES="$(ls)"
for f in $FILES
do
  if [ $rem -eq 1 ]
  then
	printf "\n^^^()()^^^" >> Output.txt
  else
	cp $2 $2$bakup
	sed '$ d' $2$bakup > $2
	rm -f $2$bakup
  fi
  git add $f
  GIT_AUTHOR_DATE=$1 GIT_COMMITTER_DATE=$1 git commit -m 'update'
  git commit -m 'update'
  git push
  # take action on each file. $f store current file name
done
git rm realwork.txt
git commit -m 'delete'
git push


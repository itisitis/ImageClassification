# Check number of arguments
if [ "$#" != "2" ]; then
	echo "Usage: bash downloadmulticategoryvideos.sh <number-of-videos-per-category> <selected-category-file-name>"
	exit 1
fi

while read line
	do
		echo "///////download multiple category///////"
		echo $1 $line
		bash downloadcategoryids.sh $1 "${line}"
		echo "///////download multiple category2///////"
		echo $1 $line
		bash downloadvideos.sh $1 ${line}
		#bash downloadvideos.sh $1 ${line}
	done < "$2"
bash generateframesfromvideos.sh videos/ frames jpg 800

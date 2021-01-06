for f in dataset/face/0*[5-9].png ; do
	echo $f
	python projector.py --image-size 256 $f
done

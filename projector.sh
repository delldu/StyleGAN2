for f in face/0*[5-9].png ; do
	echo $f
	python projector.py \
		--ckpt checkpoint/stylegan2-ffhq-config-f.pth \
		$f
done

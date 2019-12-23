#! /bin/bash
#项目根目录
basepath="/media/d9lab/data11/tomasyao/workspace/pycharm_ws/mypython"

if [ $# -ne 1 ] #有且仅有一个参数，否则退出
then
	echo "Usage: /start.sh train[|test|demo]"
	exit 1
else
	echo "starting..."
fi

if [ $1 = "dlib_align_fgnet" ]
then
	echo "dlib_align_start..."
	source activate tfg
	cd ${basepath}/util
	num_core=32
	start=0
	total=1002
	len=$(expr $total - $start)
	gap=$(expr $len / $num_core)
	((gap++))
	# echo gap - $gap
	for (( i=0; i<$num_core; i++ ))
	do
		end=$(expr $start + $gap)
		# echo $start - $end
		setsid python ./file_util.py --start=$start --end=$end > $basepath/logs/dlib_align_fgnet_thread$i 2>&1 &
		start=$end
	done
elif [ $1 = "download_ce" ]
then
	echo "download_ce_start..."
	source activate torchg
	cd ${basepath}/util
	num_core=8
	start=0
	total=53363
	len=$(expr $total - $start)
	gap=$(expr $len / $num_core)
	((gap++))
	# echo gap - $gap
	for (( i=0; i<$num_core; i++ ))
	do
		end=$(expr $start + $gap)
		# echo $start - $end
		setsid python ./file_util.py --start=$start --end=$end > $basepath/logs/download_ce_thread$i 2>&1 &
		start=$end
	done
elif [ $1 = "dlib_align_ce" ]
then
	echo "dlib_align_ce_start..."
	source activate torchg
	cd ${basepath}/util
	num_core=32
	start=0
	total=10576
	len=$(expr $total - $start)
	gap=$(expr $len / $num_core)
	((gap++))
	# echo gap - $gap
	for (( i=0; i<$num_core; i++ ))
	do
		end=$(expr $start + $gap)
		# echo $start - $end
		setsid python ./file_util.py --start=$start --end=$end > $basepath/logs/dlib_align_ce_thread$i 2>&1 &
		start=$end
	done
else
	echo "do nothing"
fi
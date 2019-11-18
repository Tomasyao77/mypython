#! /bin/bash
#项目根目录
basepath="/media/gisdom/data11/tomasyao/workspace/pycharm_ws/mypython"

if [ $# -ne 1 ] #有且仅有一个参数，否则退出
then
	echo "Usage: /start.sh train[|test|demo]"
	exit 1
else
	echo "starting..."
fi

if [ $1 = "dlib_align_morph2" ]
then
	echo "dlib_align_morph2..."
	source activate tfgpu
	num_core=40
	start=0
	total=52099
	len=$(expr $total - $start)
	gap=$(expr $len / $num_core)
	((gap++))
	# echo gap - $gap
	for (( i=0; i<$num_core; i++ ))
	do
		end=$(expr $start + $gap)
		# echo $start - $end
		setsid python ${basepath}/util/gen_txt.py --start=$start --end=$end > $basepath/logs/dlib_align_morph2_tmp_thread$i 2>&1 &
		start=$end
	done
else
	echo "do nothing"
fi
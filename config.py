from easydict import EasyDict as edict
import sys

# sys.path.append(".")
# sys.path.append("..")
cfg = edict()

cfg.batch_size = 50
cfg.BASE = "/media/d9lab/data11/tomasyao/workspace/pycharm_ws/mypython"  # 项目根目录
cfg.LOGS = cfg.BASE + "/logs"
cfg.DATASET = cfg.BASE + "/dataset"
cfg.dlib68dat = cfg.BASE + "/util/mydlib/shape_predictor_68_face_landmarks.dat"

# dataset
cfg.dataset = edict()
# morph2
cfg.dataset.morph2 = cfg.DATASET + "/morph2"
cfg.dataset.morph2_align = cfg.DATASET + "/morph2_align"
cfg.dataset.morph2_split = cfg.DATASET + "/morph2_split"
# wiki_crop
cfg.dataset.wiki_crop = cfg.DATASET + "/wiki_crop"
cfg.dataset.wiki_crop_dlibdetect = cfg.DATASET + "/wiki_crop_dlibdetect"
# FG-NET
cfg.dataset.fgnet = cfg.DATASET + "/FG-NET"
cfg.dataset.fgnet_align = cfg.DATASET + "/FG-NET_align"
cfg.dataset.fgnet_split = cfg.DATASET + "/FG-NET_split"
cfg.dataset.fgnet_leave1out = cfg.DATASET + "/FG-NET-leave1out"
cfg.dataset.fgnet_align_leave1out = cfg.DATASET + "/FG-NET_align-leave1out"
# CACD2000
cfg.dataset.cacd2000 = cfg.DATASET + "/CACD2000"
cfg.dataset.cacd2000_split = cfg.DATASET + "/CACD2000_split"

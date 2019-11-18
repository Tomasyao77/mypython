from easydict import EasyDict as edict
import sys
# sys.path.append(".")
# sys.path.append("..")
cfg = edict()

cfg.batch_size = 50
cfg.BASE = "/media/zouy/workspace/gitcloneroot/mypython"  # 项目根目录
cfg.LOGS = cfg.BASE + "/logs"
cfg.DATASET = cfg.BASE + "/dataset"
cfg.dlib68dat = cfg.BASE + "/util/mydlib/shape_predictor_68_face_landmarks.dat"

# dataset
cfg.dataset = edict()
# morph2
cfg.dataset.morph2 = cfg.DATASET + "/morph2"
cfg.dataset.morph2_split = cfg.DATASET + "/morph2_split"
cfg.dataset.morph2_align = cfg.DATASET + "/morph2_align"
# wiki_crop
cfg.dataset.wiki_crop = cfg.DATASET + "/wiki_crop"
cfg.dataset.wiki_crop_dlibdetect = cfg.DATASET + "/wiki_crop_dlibdetect"

from fastai.imports import *
from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *
from fastai.plots import *

PATH = "/Users/lijiahuan/PycharmProjects/furniture/data/"
sz = 299
bs = 28

#changed architecture from resnet34 to resnext50
#arch=resnet34
arch= resnext50
tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=8, test_name='test')
learn = ConvLearner.pretrained(arch, data, precompute=True, ps=0.5)

learn.fit(2e-3, 1)
learn.save('precomputeTrue_1.h5')

learn.precompute=False

learn.fit(2e-3, 2, cycle_len=1)
learn.save('precomputeTrue_2.h5')

learn.unfreeze()
lr=np.array([5e-5,5e-4,2e-3])
learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
learn.save('precomputeTrue_3.h5')

# predictions for valid set
log_preds,y = learn.TTA(is_test=True)
probs = np.mean(np.exp(log_preds),0)
preds = np.argmax(probs, axis=1)


# sz = 299
# bs = 18
#
# arch=resnext50
# tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
# data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs, num_workers=8, test_name='test')
# learn = ConvLearner.pretrained(arch, data, precompute=False, ps=0.5)
#
# #learn.load('precomputeFalse.h5')
# learn.unfreeze()
# lr=np.array([5e-5,5e-4,2e-3])
# #learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
# learn.load('differential.h5')
#
# learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
#
#
# log_preds,y = learn.TTA(is_test=True)
# probs = np.mean(np.exp(log_preds),0)
# preds = np.argmax(probs, axis=1)


realpred = pd.Series(preds).map(lambda x: data.classes[x])
testnames = data.test_ds.fnames

subm = pd.DataFrame()
subm['id'] = pd.Series(testnames).map(lambda x: x[5:-4]).astype(int)
subm['predicted'] = pd.Series(realpred).astype(int)

subm.set_index('id', inplace=True)
new_index = pd.Index(np.arange(1,12801,1), name="id")
subm = subm.reindex(new_index)

subm = subm.fillna(0)
subm = subm.astype(int)
subm.to_csv('subm_full_rx50.csv')
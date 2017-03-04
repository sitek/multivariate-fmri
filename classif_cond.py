import numpy as np
from nilearn import masking
from nilearn import image
from nilearn import input_data
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from nilearn.decoding import SpaceNetClassifier

# data
Xtrain = 'voice978_task005_run002_emosent_dtype_mcf_mask_smooth_mask_gms_tempfilt_maths.nii.gz'
Xtest = 'voice978_task005_run001_emosent_dtype_mcf_mask_smooth_mask_gms_tempfilt_maths.nii.gz'
ytrain = '/om/user/ysa/multivariate-fmri/ytrain.txt'
ytest = '/om/user/ysa/multivariate-fmri/ytest.txt'

# attempt one, using approach straight from nilearn documentation
# first going to create an array to index the time points of the
# incoming data
idx_all_conditions = np.tile(True, 48)
X_train = image.index_img(Xtrain, idx_all_conditions)
X_test = image.index_img(Xtest, idx_all_conditions)
y_train1 = np.genfromtxt(ytrain, dtype=float)
y_test1 = np.genfromtxt(ytest, dtype=float)

# this creates the classifier
decoder = SpaceNetClassifier()
# this trains the classifier
decoder.fit(X_train, y_train1)
# get the prediction
prediction = decoder.predict(X_test)
# compute accuray
print("% Correct attempt one: {}".format((prediction == y_test1).mean()))


# attempt two, same approach but hopefully making it an easier
# classification problem, first index everything for the conditions
idx_1_0_ytrain = np.logical_or(y_train1 == 0, y_train1 == 1)
idx_1_0_ytest = np.logical_or(y_test1 == 0, y_test1 == 1)
X_train = image.index_img(Xtrain, idx_1_0_ytrain)
X_test = image.index_img(Xtest, idx_1_0_ytest)
y_train2 = y_train[idx_1_0_ytrain]
y_test2 = y_test[idx_1_0_ytest]

# apply a new classifier
decoder.fit(X_train, y_train2)
prediction = decoder.predict(X_test)
print("% Correct attempt two: {}".format((prediction == y_test2).mean()))


# attempt 3, sklearn and niftimasker, whole brain
masker = input_data.NiftiMasker(smoothing_fwhm = None,
    standardize = False, detrend = False, low_pass = None,
    high_pass = None, t_r = None)

X_train = masker.fit_transform(Xtrain)[idx_1_0_ytrain, :]
X_test = masker.transform(Xtest)[idx_1_0_ytest, :]
# we're using ytrain1 and ytest1

clf = Pipeline([('scale', StandardScaler()),
                ('lg', linear_model.LogisticRegression(
                            penalty = 'l1',
                            solver = 'liblinear'))])

clf.fit(X_train, y_train2)
prediction = clf.predict(X_test)
print("% Correct attempt three: {}".format((prediction == y_test2).mean()))



from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import LinearSVC
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestRegressor
estimator = LinearSVC()
selector = RFE(estimator=estimator, n_features_to_select=2)
X_t = selector.fit_transform(X, y)
### 切分测试集与验证集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    test_size=0.25, random_state=0, stratify=y)
X_train_t, X_test_t, y_train_t, y_test_t = model_selection.train_test_split(X_t, y,
                                                                            test_size=0.25, random_state=0,
                                                                            stratify=y)
## 测试与验证
clf = LinearSVC()
clf_t = LinearSVC()
clf.fit(X_train, y_train)
clf_t.fit(X_train_t, y_train_t)
print("Original DataSet: test score=%s" % (clf.score(X_test, y_test)))
print("Selected DataSet: test score=%s" % (clf_t.score(X_test_t, y_test_t)))
features = data.columns.values.tolist()
features.pop(0)
print(features)
#基于学习模型的特征排序
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
scores = []
for i in range(X.shape[1]):
     score = cross_val_score(rf, X[:, i:i+1], y, scoring="r2",
                              cv=ShuffleSplit(len(X), 3, .3))
     scores.append((round(np.mean(score), 3), features[i-1]))
print(sorted(scores, reverse=True))
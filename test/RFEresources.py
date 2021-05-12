class RFE(BaseEstimator, MetaEstimatorMixin, SelectorMixin):

    def __init__(self, estimator, n_features_to_select=None, step=1,
                 estimator_params={}, verbose=0):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.estimator_params = estimator_params
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]
        if self.n_features_to_select is None:
            n_features_to_select = n_features / 2
        else:
            n_features_to_select = self.n_features_to_select

        if 0.0 < self.step < 1.0:
            step = int(self.step * n_features)
        else:
            step = int(self.step)
        if step <= 0:
            raise ValueError("Step must be >0")

        support_ = np.ones(n_features, dtype=np.bool)   # true代表就用这个特征吧
        ranking_ = np.ones(n_features, dtype=np.int)    # 值越小越好
        # Elimination
        while np.sum(support_) > n_features_to_select:
            # Remaining features
            ''' the function of np.arange
            np.arange(3) -> array([0, 1, 2]), 
            np.arange(3)[np.array([True,False,True])]-> array([0, 2])
            '''
            features = np.arange(n_features)[support_]  # 每个值代表是第几个特征；若features中第3个值为9,代表新的第3个特征是原始的第9个特征

            # Rank the remaining features
            estimator = clone(self.estimator)
            estimator.set_params(**self.estimator_params)
            if self.verbose > 0:
                print("Fitting estimator with %d features." % np.sum(support_))

            estimator.fit(X[:, features], y)  # 使用现在的特征集合来训练

            ''' the function of np.argsort
            >>> x = np.array([3, 1, 2])
            >>> np.argsort(x)
            array([1, 2, 0])
            '''
            if estimator.coef_.ndim > 1:
                ranks = np.argsort(safe_sqr(estimator.coef_).sum(axis=0))  # 将特征按找重要性从小到大来排序
            else:
                ranks = np.argsort(safe_sqr(estimator.coef_))

            # for sparse case ranks is matrix
            '''the function of np.ravel
            >>> x = np.array([[1, 2, 3], [4, 5, 6]])
            >>> print np.ravel(x)
            [1 2 3 4 5 6]

            '''
            ranks = np.ravel(ranks)

            # Eliminate the worse features
            '''
            >>> np.array([89,20,100])[np.array([2,0,1])]
            >>> array([100,  89,  20])
            >>> np.logical_not(3)
            False
            >>> np.logical_not([True, False, 0, 1])
            array([False,  True,  True, False], dtype=bool)
            '''
            threshold = min(step, np.sum(support_) - n_features_to_select) # 该删除多少特征
            support_[features[ranks][:threshold]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(X[:, support_], y)
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        return self
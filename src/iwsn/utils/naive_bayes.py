# -*- coding: utf-8 -*-
# @Author: Yajing Zhang
# @Emial:  amberimzyj@qq.com
# @Date:   2020-04-21 15:27:10
# @Last Modified by:   Yajing Zhang
# @Last Modified time: 2020-04-25 13:16:23
# @License: MIT LICENSE

import numpy as np
from sklearn import naive_bayes
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


class NavieBayes(object):

    def __init__(self, bayes_type: str):
        if not bayes_type in ['MultinomialNB']:
            raise Exception(f'The {bayes_type} is not allowed')

        self.type_ = bayes_type
        self._preprocess_func = getattr(self, bayes_type.lower())
        self._enc = None

        self._model = getattr(naive_bayes, bayes_type)()

    def fit(self, train_X_index: np.ndarray, train_X: np.ndarray, train_y: np.ndarray, partial: False):
        self._train_X_index = train_X_index
        self._train_X = train_X
        self._train_y = train_y
        self._preprocess_func(train_X)
        train_X = self._encode_data(train_X)

        if partial:
            self._model.partial_fit(train_X, train_y, (0, 1))
        else:
            self._model.fit(train_X, train_y)
        self.fitted_ = True

    def predict(self, X: np.ndarray):
        check_is_fitted(self)
        if self._enc is not None:
            X = self._enc.transform(X)

        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray):
        check_is_fitted(self)
        if self._enc is not None:
            X = self._enc.transform(X)

        return self._model.predict_proba(X)

    def multinomialnb(self, X):
        if self._enc is None:
            self._enc = OneHotEncoder(sparse=False)
            self._enc.fit(X)

    def _encode_data(self, X):
        if self._enc is None:
            return X

        return self._enc.transform(X)

    def condition_prob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if self._enc is not None:
            X = self._enc.transform(X)

        all_feature_log_prob = np.dot(X, self._model.feature_log_prob_.T)
        sorter = np.argsort(self._model.classes_)
        y = sorter[np.searchsorted(self._model.classes_, y, sorter=sorter)]
        condition_prob = np.exp(all_feature_log_prob[np.arange(len(X)), y])

        return condition_prob

    def class_prior_prob(self, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        sorter = np.argsort(self._model.classes_)
        y = sorter[np.searchsorted(self._model.classes_, y, sorter=sorter)]
        class_prior_prob = np.exp(self._model.class_log_prior_[y])

        return class_prior_prob

    def feature_prior_prob(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if self._enc is not None:
            X = self._enc.transform(X)

        jll = self._model._joint_log_likelihood(X)
        feature_prior_prob = np.sum(np.exp(jll), axis=1)

        return feature_prior_prob

    def join_prob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        condition_prob = self.condition_prob(X, y)
        class_prior_prob = self.class_prior_prob(y)
        join_prob = condition_prob * class_prior_prob

        return join_prob

    def posterior_prob(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.condition_prob(X, y)

    def mutual_information(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        join_prob = self.join_prob(X, y)
        class_prior_prob = self.class_prior_prob(y)
        feature_prior_prob = self.feature_prior_prob(X)
        mi = join_prob * \
            np.log2(join_prob / class_prior_prob / feature_prior_prob)

        return mi

    def chi_square_test(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        n_samples = len(X)

        npq = n_samples * self.join_prob(X, y)
        epq = n_samples * self.class_prior_prob(y) * self.feature_prior_prob(X)

        return (npq - epq) ** 2 / epq

    def select(self, X: np.ndarray, y: np.ndarray, metric: str) -> np.ndarray:
        if not metric in ['posterior_prob', 'mutual_information', 'chi_square_test']:
            raise ValueError('The spectified metric not supported.')

        metrics = np.expand_dims(self._scale_data(
            getattr(self, metric)(X, y)), 0)
        indexs = np.expand_dims(self._train_X_index, 0)
        # metrics[metrics <= thresh] = 0.
        indexed_metrics = np.concatenate([indexs, metrics]).T
        indexed_metrics = -np.sort(-indexed_metrics, axis=0)

        unique_index = np.unique(indexed_metrics[:, 0], return_index=True)[1]
        unique_index.sort()
        indexed_metrics = indexed_metrics[unique_index]

        return indexed_metrics

    def _scale_data(self, data: np.ndarray):
        return (data - data.min()) / (data.max() - data.min())


if __name__ == '__main__':
    # train_X = np.random.randint(1, 6, (10, 2))
    # train_y = np.random.randint(1, 4, 10)

    train_X = np.array([
        [3, 1],
        [4, 4],
        [3, 5],
        [4, 1],
        [2, 2],
        [4, 5],
        [1, 3],
        [4, 5],
        [5, 1],
        [5, 3]], dtype='int')
    train_y = np.array([1, 1, 1, 3, 1, 3, 3, 2, 1, 3], dtype='int')

    test_X = train_X[0:2]
    test_y = train_y[0:2]

    nb = NavieBayes('MultinomialNB')
    nb.fit(train_X, train_y)

    condition_prob = nb.condition_prob(test_X, test_y)
    class_prior_prob = nb.class_prior_prob(test_y)
    feature_prior_prob = nb.feature_prior_prob(test_X)
    join_prob = nb.join_prob(test_X, test_y)
    cmpt_condition_prob = nb.cmpt_condition_prob(test_X, test_y)
    cmpt_class_prior_prob = nb.cmpt_class_prior_prob(test_y)
    cmpt_feature_prior_prob = nb.cmpt_feature_prior_prob(test_X)
    cmpt_join_prob = nb.cmpt_join_prob(test_X, test_y)
    post_prob = nb.posterior_prob(test_X, test_y)
    mi = nb.mutual_information(test_X, test_y)
    csq = nb.chi_square_test(test_X, test_y)
    max_metric = nb.max_metric(test_X, test_y)

    print('Train X: ', train_X)
    print('Train y: ', train_y)
    print('Test X: ', test_X)
    print('Test y: ', test_y)
    print('condition_prob: ', condition_prob)
    print('class_prior_prob: ', class_prior_prob)
    print('feature_prior_prob: ', feature_prior_prob)
    print('join_prob: ', join_prob)
    print('cmpt_condition_prob: ', cmpt_condition_prob)
    print('cmpt_class_prior_prob: ', cmpt_class_prior_prob)
    print('cmpt_feature_prior_prob: ', cmpt_feature_prior_prob)
    print('cmpt_join_prob: ', cmpt_join_prob)
    print('posterior_prob: ', post_prob)
    print('mutual_information: ', mi)
    print('chi_square_test:', csq)
    print('max_metric:', max_metric)

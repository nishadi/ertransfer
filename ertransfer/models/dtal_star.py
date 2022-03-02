import logging
import os

import sklearn
import torch.nn as nn
from torch.autograd import Function
import deepmatcher as dm


class GradReverse(Function):

  @staticmethod
  def forward(ctx, x):
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * 0.5

    return output


def grad_reverse(x):
  return GradReverse.apply(x)


class DANNModel(nn.Module):
  """
  A neural network model for transfer learning in entity matching.

  Refer to
  'Low-resource Deep Entity Resolution with Transfer and Active Learning'
  for details on this model.

      matching_classifier (string or :class:`Classifier` or callable):
          The neural network to perform match / mismatch classification
          for matching classifier.
          Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#classifier-can-be-set-to-one-of-the-following:>`__.
          Defaults to '2-layer-highway', i.e., use a two layer highway
          network followed by a softmax layer for classification.
      matching_hidden_size (int):
          The hidden size to use for the `attr_summarizer` and the
          `classifier` modules for matching classifier, if they are string
          arguments. If a module or :attr:`callable` input is specified
          for a component, this argument is ignored for that component.
      dataset_classifier (string or :class:`Classifier` or callable):
          The neural network to perform match / mismatch classification
          for data set classifier.
          Options `listed here <https://nbviewer.jupyter.org/github/sidharthms/deepmatcher/blob/master/examples/matching_models.ipynb#classifier-can-be-set-to-one-of-the-following:>`__.
          Defaults to '2-layer-highway', i.e., use a two layer highway
          network followed by a softmax layer for classification.
      dataset_hidden_size (int):
          The hidden size to use for the `attr_summarizer` and the
          `classifier` modules or data set classifier, if they are string
          arguments. If a module or :attr:`callable` input is specified for
          a component, this argument is ignored for that component.

  """

  def __init__(self,
               matching_classifier='2-layer-highway',
               matching_hidden_size=300,
               dataset_classifier='2-layer-highway',
               dataset_hidden_size=300):
    super(DANNModel, self).__init__()
    self.matching_classifier = dm.Classifier(
      dm.modules.Transform(matching_classifier,
                           hidden_size=matching_hidden_size))
    self.dataset_classifier = dm.Classifier(
      dm.modules.Transform(dataset_classifier,
                           hidden_size=dataset_hidden_size))

  def forward(self, feature):
    reverse_feature = grad_reverse(feature)
    class_output = self.matching_classifier(feature)
    domain_output = self.dataset_classifier(reverse_feature)

    return class_output, domain_output


class DTAL_star():
  """
  Implementation of the DANN model for transfer learning in entity resolution.

  Refer to
  'Low-resource Deep Entity Resolution with Transfer and Active Learning'
  for details on this model.
  However, it is important to note that we have not implemented the active
  learning part of the DTAL algorithm.


  """

  def __init__(self,
               basepath='data/dtal/',
               epochs=20,
               batch_size=16,
               pos_neg_ratio=3):
    self.epochs = epochs
    self.batch_size = batch_size
    self.model = dm.MatchingModel(classifier=DANNModel)
    self.base_path = basepath
    self.pos_neg_ratio = pos_neg_ratio

  def _preprocess_data(self, X, X_name):

    # Split data into train, test, and valid
    X_train_path = '{}/{}-train.csv'.format(self.base_path, X_name)
    X_test_path = '{}/{}-test.csv'.format(self.base_path, X_name)
    X_valid_path = '{}/{}-valid.csv'.format(self.base_path, X_name)

    if not os.path.isfile(X_train_path):

      if not os.path.exists(self.base_path):
        os.makedirs(self.base_path)

      X.rename(columns={'_id': 'id'}, inplace=True)
      if 'left_id' in X.columns:
        del X['left_id']
        del X['right_id']
      X_train, X_test, _, _ = \
        sklearn.model_selection.train_test_split(X,
                                                 X['label'],
                                                 test_size=0.4,
                                                 train_size=0.6,
                                                 random_state=0,
                                                 shuffle=True,
                                                 stratify=X['label'])
      X_test, X_valid, _, _ = \
        sklearn.model_selection.train_test_split(X_test,
                                                 X_test['label'],
                                                 test_size=0.5,
                                                 train_size=0.5,
                                                 random_state=0,
                                                 shuffle=True,
                                                 stratify=X_test['label'])
      X_train.to_csv(X_train_path, index=False)
      X_test.to_csv(X_test_path, index=False)
      X_valid.to_csv(X_valid_path, index=False)

  def fit(self, Xs=None, Xt=None, Xs_name=None, Xt_name=None):

    # Preprocess data
    self._preprocess_data(Xs, Xs_name)
    self._preprocess_data(Xt, Xt_name)
    train, validation, test = dm.data.process(
      path=self.base_path,
      train='{}-train.csv'.format(Xs_name),
      validation='{}-test.csv'.format(Xs_name),
      test='{}-train.csv'.format(Xt_name))

    print('Train data set shape:', len(train.examples))
    print('Test data set shape:', len(test.examples))

    # Run model
    print(self.base_path + Xs_name + Xt_name + 'hybrid-model.pth')
    self.model.run_train(
      train,
      validation,
      test,
      # epochs=self.epochs,
      epochs=1,
      batch_size=self.batch_size,
      # best_save_path=self.base_path + Xs_name + Xt_name + 'hybrid-model.pth',
      best_save_path='hybrid-model.pth',
      pos_neg_ratio=self.pos_neg_ratio)

    # Evaluate model

    f1, stats = self.model.run_eval(test)
    logging.info('F1 score : {}'.format(f1))
    logging.info('Precision : {}'.format(stats.precision()))
    logging.info('Recall : {}'.format(stats.recall()))

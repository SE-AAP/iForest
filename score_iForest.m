%score_iForest  function to compute isolationForest scores based on a trained model and some testing data
%
%  Syntax
%
% scores = score_iForest(model,test_data)
%
%  Input
%
% - model:     a trained iForest model (output of train_iForest)
% - test_data: some testing data compatible with the data used to train the model. 
%              Can be a data matrix or an object of class NormFeatures (or a subclass).
%
%  Output
%
% - scores: the anomaly scores

% G. Rilling 2017
% CEA/DRT/LIST/DM2I/LADIS
function [val,tree_scores] = score_iForest(iForest,data_test)

if ~isa(data_test,'NormFeatures') 
  if isfield(iForest,'data') && isa(iForest.data,'NormFeatures')
    tmp = copy(iForest.data);
    tmp.data = data_test;
    tmp.normalization_params.type = 'manual';
    tmp.manual_normalization = iForest.data.normalization;
    data_test = tmp.features_norm;
  end
else
  if isfield(iForest,'data') && isa(iForest.data,'NormFeatures') && ~isequal(data_test.normalization,iForest.data.normalization)
    warning('the normalization of the provided data is not consistent with the normalization of the training data');
  end
  data_test = data_test.features_norm;
end

tree_scores = IsolationEstimation(data_test.',iForest).';
val = -mean( tree_scores, 1);
end

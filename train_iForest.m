%train_iForest  function to train an isolationForest model
%
%  Syntax
%
% [model,training_anomaly_scores] = train_iForest(data,...,PropertyName,PropertyValue,...)
%
%  Input
%
% - data: some training data. Can be matrix of features (stored as columns) or 
%         an object deriving from the NormFeatures class.
% - PropertyName,PropertyValue pairs: set the values of IsolationForest parameters
%
% Available properties:
% - 'NumTree':                       the number of trees in the iForest
%                                    [ Default: 100 ]
% - 'NumSub':                        the size of each random subsample used to build a tree
%                                    [ Default: 256 ]
% - 'NumDim':                        the number of data dimensions used in each tree
%                                    [ Default: size(data,1), i.e. all dimensions ]
% - 'rseed':                         the seed of the random number generator (outdated implementation, may not be future proof)
%                                    [ Default: 0 ]
%
%  Output
%
% - model:                           the trained iForest model
% - training_anomaly_scores:         the anomaly scores for the training data

% G. Rilling 2018
% CEA/DRT/LIST/DM2I/LADIS

function [iForest,training_score] = train_iForest(data,varargin)

% rem: unless the normalization mixes data dimensions, it will have zero effect
if isa(data,'NormFeatures')
  nf = data;
  data = nf.features_norm;
else
  nf = NormFeatures(data,'normalization_params',{'type','none'}); % this means no normalization (since it has no impact on IsolationForest anyway)
end

parser = inputParser;
parser.addParameter('NumTree',100);
parser.addParameter('NumSub',256);
parser.addParameter('NumDim',size(data,1));
parser.addParameter('rseed',0);

parser.parse(varargin{:});

iForest.NumTree = ceil(parser.Results.NumTree); % use ceil in case value is not integer (might happen with some autotuning testing parameters on a non-integer grid)
iForest.NumSub  = ceil(parser.Results.NumSub);
iForest.NumDim  = ceil(parser.Results.NumDim);
iForest.rseed   = parser.Results.rseed;

iForest = IsolationForest(data.', iForest.NumTree, iForest.NumSub, iForest.NumDim, iForest.rseed);
iForest.data = nf;

if nargout > 1
  training_score = score_iForest(iForest,nf);
end

classdef chDropoutLayer < nnet.layer.Layer    

  properties
    probability
  end

  properties (Learnable)
  end

  methods
    function layer = chDropoutLayer(name,probability)
      layer.probability = single(probability);                  
      layer.Name        = name;                  
    end

    function X = predict(layer,X)
      X = X;
%       opd_ind_drp = randperm(size(X,3));
%       opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(layer.probability)));
% 
%       X(:,:,opd_ind_drp,:,:) = 0;      
    end

    function X = forward(layer,X)
      opd_ind_drp = randperm(size(X,3));
      opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(layer.probability)));

      X(:,:,opd_ind_drp,:,:) = 0;      
    end
    
  end
end
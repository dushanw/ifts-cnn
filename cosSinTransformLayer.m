classdef cosSinTransformLayer < nnet.layer.Layer    

  properties
    Nopd
    Nk
    opd
    k
    A_cos
    A_sin
  end

  properties (Learnable)
  end

  methods
    function layer = cosSinTransformLayer(name,opd,k)
      layer.Nopd    = length(opd);
      layer.Nk      = length(k);      
      layer.opd     = reshape(opd  ,[1  1  1        layer.Nopd]);
      layer.k       = reshape(k    ,[1  1  layer.Nk 1         ]);
      layer.A_cos   = cos(layer.opd.*layer.k);
      layer.A_sin   = sin(layer.opd.*layer.k);
      
      layer.Name        = name;                  
    end

    function X = predict(layer,X)
      % X will have input dimentions [y x opd b]
                                 % y         x         k opd       b   
      X             = reshape(X  ,[size(X,1) size(X,2) 1 size(X,3) size(X,4)]);
      %Z             = sqrt(sum(Z.*layer.A_cos,4).^2 + sum(Z.*layer.A_sin,4).^2)/2*pi;
      X             = (sum(X.*layer.A_cos,4).^2 + sum(X.*layer.A_sin,4).^2);
      X             = reshape(X  ,[size(X,1) size(X,2) layer.Nk size(X,5)]);
    end
    
  end
end
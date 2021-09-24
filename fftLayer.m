classdef fftLayer < nnet.layer.Layer    

  properties
    Nk
  end

  properties (Learnable)
  end

  methods
    function layer = fftLayer(name,k)    
      layer.Nk     = length(k);
      layer.Name   = name;
    end

    function X = predict(layer,X)
      % X will have input dimentions [y x opd b]
      X   = abs(fft(X,[],3));
      X   = X(:,:,2:layer.Nk+1,:);
    end
    
  end
end
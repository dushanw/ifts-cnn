classdef fftLayer < nnet.layer.Layer    

  properties
    Nk
    visIdx
  end

  properties (Learnable)
  end

  methods
    function layer = fftLayer(name,k)    
      layer.Nk     = length(k);
      layer.visIdx = find(2*pi./k>300 & 2*pi./k<800)+1;
      layer.Name   = name;
    end

    function X = predict(layer,X)
      % X will have input dimentions [y x opd b]
      X   = abs(fft(X,[],3));
      % X   = X(:,:,2:layer.Nk+1,:);
      X   = X(:,:,layer.visIdx,:);
    end
    
  end
end
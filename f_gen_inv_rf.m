function lgraph = f_gen_inv_rf(pram)

  size_In     = [pram.Ny pram.Nx pram.N_opdCompresed];

  net0_dncnn  = denoisingNetwork('dncnn');
  lgraph      = layerGraph(net0_dncnn.Layers);

  lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer(size_In,'Name','InputLayer','Normalization','none'));
  lgraph = replaceLayer(lgraph,'Conv1',convolution2dLayer([3 3],64,'Name','Conv1','NumChannels',pram.N_k,'Padding',[1 1 1 1],'stride',[1  1]));        
  
  lgraph = replaceLayer(lgraph,'Conv16',convolution2dLayer([3 3],pram.N_k,'Name','Conv16','Padding',[1 1 1 1],'stride',[1  1]));
  lgraph = replaceLayer(lgraph,'Conv17',convolution2dLayer([3 3],pram.N_k,'Name','Conv17','Padding',[1 1 1 1],'stride',[1  1]));
  lgraph = replaceLayer(lgraph,'Conv18',convolution2dLayer([3 3],pram.N_k,'Name','Conv18','Padding',[1 1 1 1],'stride',[1  1]));
  lgraph = replaceLayer(lgraph,'Conv19',convolution2dLayer([3 3],pram.N_k,'Name','Conv19','Padding',[1 1 1 1],'stride',[1  1]));
  lgraph = replaceLayer(lgraph,'Conv20',convolution2dLayer([3 3],pram.N_k,'Name','Conv20','Padding',[1 1 1 1],'stride',[1  1]));
  lgraph = replaceLayer(lgraph,'BNorm16',batchNormalizationLayer('Name','BNorm16'));      
  lgraph = replaceLayer(lgraph,'BNorm17',batchNormalizationLayer('Name','BNorm17'));      
  lgraph = replaceLayer(lgraph,'BNorm18',batchNormalizationLayer('Name','BNorm18'));      
  lgraph = replaceLayer(lgraph,'BNorm19',batchNormalizationLayer('Name','BNorm19'));
  
  % lgraph = addLayers(lgraph,dropoutLayer(1 - pram.compression,'Name','Drop1'));
  % lgraph = addLayers(lgraph,chDropoutLayer('Drop1',1-pram.compression));
  lgraph = addLayers(lgraph,cosSinTransformLayer('cosSinTr1',pram.opdCompressed,pram.k));
  % lgraph = addLayers(lgraph,fftLayer('fft1',pram.kAll));
  lgraph = disconnectLayers(lgraph,'InputLayer','Conv1');
  lgraph = connectLayers(lgraph,'InputLayer','cosSinTr1');  
  lgraph = connectLayers(lgraph,'cosSinTr1','Conv1');
  % lgraph = connectLayers(lgraph,'Drop1','fft1');
  % lgraph = connectLayers(lgraph,'fft1','Conv1');

  lgraph = addLayers(lgraph,additionLayer(2,'Name','add_2_5'));
  lgraph = addLayers(lgraph,additionLayer(2,'Name','add_5_10'));
  lgraph = addLayers(lgraph,additionLayer(2,'Name','add_10_15'));
  lgraph = addLayers(lgraph,additionLayer(2,'Name','add_16_19'));

  lgraph = disconnectLayers(lgraph,'BNorm5','ReLU5');
  lgraph = disconnectLayers(lgraph,'BNorm10','ReLU10');
  lgraph = disconnectLayers(lgraph,'BNorm15','ReLU15');
  lgraph = disconnectLayers(lgraph,'BNorm19','ReLU19');

  lgraph = connectLayers(lgraph,'BNorm5','add_2_5/in1');
  lgraph = connectLayers(lgraph,'ReLU2','add_2_5/in2');
  lgraph = connectLayers(lgraph,'add_2_5','ReLU5');

  lgraph = connectLayers(lgraph,'BNorm10','add_5_10/in1');
  lgraph = connectLayers(lgraph,'ReLU5','add_5_10/in2');
  lgraph = connectLayers(lgraph,'add_5_10','ReLU10');

  lgraph = connectLayers(lgraph,'BNorm15','add_10_15/in1');
  lgraph = connectLayers(lgraph,'ReLU10','add_10_15/in2');
  lgraph = connectLayers(lgraph,'add_10_15','ReLU15');

  lgraph = connectLayers(lgraph,'BNorm19','add_16_19/in1');
  lgraph = connectLayers(lgraph,'ReLU16','add_16_19/in2');
  lgraph = connectLayers(lgraph,'add_16_19','ReLU19');        
end






% 2021-09-20 by Dushan N. Wadduwage
% main code
clear all; clc

%% model parameters
pram        = f_pram_init();

%% generate training + testing data (contans the fwd model)
pram.Nb     = 2^12;
pram.N_mb   = 2^3;
pram.useGPU = 0;                              % some function on the fwd model doesn't support gpus
[Data Data0 pram] = f_gen_allData_beads(pram);
pram.useGPU = gpuDeviceCount>1 ;

pram.kAll       = pram.k;      
pram.lambdaAll  = pram.lambda;
inds_spec_vis   = find(pram.lambda>300 & pram.lambda<800);
Data_x          = Data;
Data.Str        = Data.Str (:,:,inds_spec_vis,:,:,:);
Data.Sval       = Data.Sval(:,:,inds_spec_vis,:,:,:);
pram.k          = pram.k(inds_spec_vis);
pram.N_k        = length(pram.k);
pram.lambda     = pram.lambda(inds_spec_vis);
  
%% generate inv model
lgraph          = f_gen_inv(pram);

%% train inv model
Data.Ival   = (Data.Ival - Data.Ival(:,:,:,1,:,:))/100;
Data.Itr    = (Data.Itr  - Data.Itr (:,:,:,1,:,:))/100;
Data.Sval   = Data.Sval./100;
Data.Str    = Data.Str ./100;

pram.miniBatchSize = 2; 
trOptions   = f_set_training_options(pram,squeeze(Data.Ival),squeeze(Data.Sval));
net         = trainNetwork(squeeze(Data.Itr),squeeze(Data.Str),lgraph,trOptions);

%% test inv model
opd_ind_drp = randperm(size(Data.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-pram.compression)));

I           = Data.Ival;
I(:,:,:,opd_ind_drp,:,:) = 0;

for i=1:size(I,6)
  i
  Shat_val(:,:,:,i)    = predict(net,squeeze(I(:,:,:,:,:,i)));
end

Shat        = Shat_val;
S0          = squeeze(Data.Sval);

MSE = mean((Shat_val(:) - S0(:)).^2)
for i=1:pram.N_mb
  [y1 x1]   = find(mean(I(:,:,:,:,:,i).^2,4)==max(max(mean(I(:,:,:,:,:,i).^2,4))));
  
  figure;
  subplot(1,2,1);imagesc(mean(S0(:,:,:,i),3));axis image;axis off;colorbar
    hold on; plot(x1,y1,'X','MarkerSize',20);hold off
  subplot(1,2,2);plot(rescale(squeeze(Shat(y1,x1,:,i))));hold on
                 plot(rescale(squeeze(S0(y1,x1,:,i))));hold on
  
  % exportgraphics(gcf,[resDir sprintf('fig_ifts_%d.png',i)],'Resolution',330)
  % saveas(gcf,[resDir sprintf('fig_ifts_%d.fig',i)])
end

%% compression tests on independent test dataset
pram_temp = pram;
pram_temp.Nb = 16;
pram_temp.useGPU = 0;
[Data_temp Data0_temp pram_temp] = f_gen_allData_beads(pram_temp);
Data_temp.Ival   = (Data_temp.Ival - Data_temp.Ival(:,:,:,1,:,:))/100;
Data_temp.Itr    = (Data_temp.Itr  - Data_temp.Itr (:,:,:,1,:,:))/100;
Data_temp.Sval   = Data_temp.Sval./100;
Data_temp.Str    = Data_temp.Str ./100;
Data_temp.Str    = Data_temp.Str (:,:,inds_spec_vis,:,:,:);
Data_temp.Sval   = Data_temp.Sval(:,:,inds_spec_vis,:,:,:);

test_compression = 0.2;

S0          = squeeze(Data_temp.Sval);

opd_ind_drp = randperm(size(Data_temp.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-test_compression)));
I           = Data_temp.Ival;
I(:,:,:,opd_ind_drp,:,:) = 0;

%Shat_val    = predict(net_chDrop,squeeze(I));
%Shat_val    = predict(net_normalDrop,squeeze(I));
%Shat_val    = predict(net_chDrop_p1,squeeze(I));
for i=1:size(I,6)% for new network with cosSinTrlayer
  i
  Shat_val(:,:,:,i)    = predict(net,squeeze(I(:,:,:,:,:,i)));
end


Shat        = Shat_val;
MSE = mean((Shat(:) - S0(:)).^2)

%% train for compression 0.1
new_tr_compression = 0.1;
lgraph = layerGraph(net_chDrop_p5);
lgraph = replaceLayer(lgraph,'Drop1',chDropoutLayer('Drop1',1-new_tr_compression));

net_chDrop_p1 = trainNetwork(squeeze(Data.Itr),squeeze(Data.Str),lgraph,trOptions);

test_compression_list = [1 0.5 0.2 0.1];
for i=1:length(test_compression_list)
  test_compression = test_compression_list(i);
  
  opd_ind_drp = randperm(size(Data_temp.Ival,4));
  opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-test_compression)));
  I           = Data_temp.Ival;
  I(:,:,:,opd_ind_drp,:,:) = 0;
  
  Shat_test     = predict(net_nDrop_p5, squeeze(I));
  Shat_chDrop_p5    = predict(net_chDrop_p5,squeeze(I));
  Shat_chDrop_p1    = predict(net_chDrop_p1,squeeze(I));
  
  MSE_nDrop_p5(i,1) = mean((Shat_test (:) - S0(:)).^2);
  MSE_chDrop_p5(i,1)= mean((Shat_chDrop_p5(:) - S0(:)).^2);
  MSE_chDrop_p1(i,1)= mean((Shat_chDrop_p1(:) - S0(:)).^2);
end

table(MSE_nDrop_p5,MSE_chDrop_p5,MSE_chDrop_p1)

imagesc(log10([MSE_nDrop_p5,MSE_chDrop_p5,MSE_chDrop_p1]));colorbar

%% Apply to real data
I           = (Data.Itst - Data.Itst(:,:,:,1,:,:))/100;

test_compression = 0.2;
opd_ind_drp = randperm(size(Data.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-test_compression)));

I(:,:,:,opd_ind_drp,:,:) = 0;


Shat_test     = predict(net, squeeze(I));

Shat_test(Shat_test<0)=0;
%plot(reshape(Shat_test,[128*128 142])')
%figure;plot(squeeze(I(97,76,:,:,:,:)))
figure;plot(squeeze(pram.lambda),rescale(squeeze(Shat_test(97,67,:))));

%% using fft based method
pram_temp             = f_pram_init();
pram_temp.Nb          = 16;
pram_temp.useGPU      = 0;                              % some function on the fwd model doesn't support gpus
[Data_temp Data0_temp pram_temp]= f_gen_allData_beads(pram_temp);
pram_temp.useGPU      = gpuDeviceCount>1 ;

Itst        = Data0.Y_beads1A(64:64+pram.Ny-1,320:320+pram.Nx-1,:);
I           = (Itst - Itst(:,:,1))/100;
Shatfft     = abs(fft(I,[],3));
Shatfft     = Shatfft(:,:,2:pram_temp.N_k+1);

figure;plot(squeeze(I(97,76,:,:,:,:)))
plot(squeeze(Shatfft(97,76,:)));

figure;plot(squeeze(pram_temp.lambda),rescale(squeeze(Shatfft(97,67,:))));xlim([300 800])


%% make and train a pixel-wise model
lgraph = layerGraph(net);
lgraph = replaceLayer(lgraph,'InputLayer',imageInputLayer([3 3 256],'Name','InputLayer','Normalization','none'));
%net_pxWise = assembleNetwork(lgraph);
%Shat_pxWise = activations(net_pxWise,squeeze(I(:,:,:,:,:,1)),'Conv20');
Data_pxwise.Itr   = [];
Data_pxwise.Str   = [];
Data_pxwise.Ival  = [];
Data_pxwise.Sval  = [];
for i=1:3:size(Data.Itr,1)-3
  i
  for j=1:3:size(Data.Itr,1)-3
    % tr
    I_small = Data.Itr(i:i+2,j:j+2,:,:,:,:);
    S_small = Data.Str(i:i+2,j:j+2,:,:,:,:);
    inds    = find(squeeze(sum(abs(I_small(2,2,:,:,:,:)),4))>10);
    
    Data_pxwise.Itr = cat(6,Data_pxwise.Itr,I_small(:,:,:,:,:,inds));
    Data_pxwise.Str = cat(6,Data_pxwise.Str,S_small(:,:,:,:,:,inds));

    % val
    I_small = Data.Ival(i:i+2,j:j+2,:,:,:,:);
    S_small = Data.Sval(i:i+2,j:j+2,:,:,:,:);
    inds    = find(squeeze(sum(abs(I_small(2,2,:,:,:,:)),4))>10);
    
    Data_pxwise.Ival = cat(6,Data_pxwise.Ival,I_small(:,:,:,:,:,inds));
    Data_pxwise.Sval = cat(6,Data_pxwise.Sval,S_small(:,:,:,:,:,inds));
  end
end

pram.miniBatchSize = 1024; 
trOptions   = f_set_training_options(pram,squeeze(Data_pxwise.Ival),squeeze(Data_pxwise.Sval));
net         = trainNetwork(squeeze(Data_pxwise.Itr),squeeze(Data_pxwise.Str),lgraph,trOptions);

%% test pixel-wise model on validation data
opd_ind_drp = randperm(size(Data_pxwise.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-pram.compression)));
I           = Data_pxwise.Ival;
I(:,:,:,opd_ind_drp,:,:) = 0;
Shat_val    = predict(net,squeeze(I));
Shat        = Shat_val;
S0          = squeeze(Data_pxwise.Sval);
MSE         = mean((Shat_val(:) - S0(:)).^2)
for i=1:16
  figure;
  plot(rescale(squeeze(Shat(2,2,:,randi(size(Shat,4))))));hold on
  plot(rescale(squeeze(S0  (2,2,:,randi(size(Shat,4))))));hold off
end

%% test pixel-wise model on real data
opd_ind_drp = randperm(size(Data_pxwise.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-pram.compression)));
I           = (Data.Itst - Data.Itst(:,:,:,1,:,:))/100;
I(:,:,:,opd_ind_drp,:,:) = 0;

Shat_test   = activations(net,squeeze(I),'Conv20');% this didn't work due to the receptive field problem
Shat_test(Shat_test<0)=0;
%plot(reshape(Shat_test,[128*128 142])')
%figure;plot(squeeze(I(97,76,:,:,:,:)))
figure;plot(squeeze(pram.lambda),rescale(squeeze(Shat_test(95,78,:))));

%on ~ single pixel
Shat_test   = activations(net,squeeze(I(94:96,77:79,:,:)),'Conv20');
Shat_test(Shat_test<0)=0;
figure;
plot(squeeze(pram.lambda),rescale(squeeze(Shat_test(2,2,:))));hold on
plot(squeeze(pram_temp.lambda),rescale(squeeze(Shatfft(97,67,:))));xlim([300 800]);hold off


%% make and train randomly selected fixed OPD case
opd_inds_rf = randperm(size(pram.opd,4));
opd_inds_rf = sort(opd_inds_rf(1:round(length(opd_inds_rf)*(pram.compression))));

pram.opdCompressed  = pram.opd(:,:,:,opd_inds_rf); 
pram.N_opdCompresed = size(pram.opdCompressed,4);
lgraph              = f_gen_inv_rf(pram);

pram.miniBatchSize  = 4; 
trOptions           = f_set_training_options(pram,squeeze(Data.Ival(:,:,:,opd_inds_rf,:,:)),...
                                                  squeeze(Data.Sval));
net                 = trainNetwork(squeeze(Data.Itr(:,:,:,opd_inds_rf,:,:)),...
                                   squeeze(Data.Str),...
                                   lgraph,...
                                   trOptions);
%% test randomly selected fixed OPD case on validation data
I           = Data.Ival(:,:,:,opd_inds_rf,:,:);
for i=1:size(I,6)
  i
  Shat_val_rf(:,:,:,i)    = predict(net,squeeze(I(:,:,:,:,:,i)));
end
Shat        = Shat_val_rf;
S0          = squeeze(Data.Sval);

MSE = mean((Shat(:) - S0(:)).^2)
for i=1:pram.N_mb
  [y1 x1]   = find(mean(I(:,:,:,:,:,i).^2,4)==max(max(mean(I(:,:,:,:,:,i).^2,4))));
  
  figure;
  subplot(1,2,1);imagesc(mean(S0(:,:,:,i),3));axis image;axis off;colorbar
    hold on; plot(x1,y1,'X','MarkerSize',20);hold off
  subplot(1,2,2);plot(rescale(squeeze(Shat(y1,x1,:,i))));hold on
                 plot(rescale(squeeze(S0(y1,x1,:,i))));hold on
end

%% test randomly selected fixed OPD case on real data
I         = (Data.Itst - Data.Itst(:,:,:,1,:,:))/1000;
I         = I(:,:,:,opd_inds_rf_p2,:,:);
Shat_test = predict(net_rf_p2, squeeze(I));

Shat_test(Shat_test<0)=0;
%plot(reshape(Shat_test,[128*128 142])')
%figure;plot(squeeze(I(97,76,:,:,:,:)))
figure;plot(squeeze(pram.lambda),rescale(squeeze(Shat_test(95,78,:))));hold on
plot(squeeze(pram_temp.lambda),rescale(squeeze(Shatfft(95,78,:))));xlim([300 800]);hold off











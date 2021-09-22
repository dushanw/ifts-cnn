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

inds_spec_vis = find(Data0.specBank_calbed.lambda>300 & Data0.specBank_calbed.lambda<800);
Data_x      = Data;
Data.Str    = Data.Str (:,:,inds_spec_vis,:,:,:);
Data.Sval   = Data.Sval(:,:,inds_spec_vis,:,:,:);
pram.N_k    = length(inds_spec_vis);

%% generate inv model
lgraph      = f_gen_inv(pram);

%% train inv model
Data.Ival   = (Data.Ival - Data.Ival(:,:,:,1,:,:))/100;
Data.Itr    = (Data.Itr  - Data.Itr (:,:,:,1,:,:))/100;
Data.Sval   = Data.Sval./100;
Data.Str    = Data.Str ./100;

trOptions   = f_set_training_options(pram,squeeze(Data.Ival),squeeze(Data.Sval));
net         = trainNetwork(squeeze(Data.Itr),squeeze(Data.Str),lgraph,trOptions);

%% test inv model
opd_ind_drp = randperm(size(Data.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-pram.compression)));

I           = Data.Ival;
I(:,:,:,opd_ind_drp,:,:) = 0;

%Shat_val    = predict(net,squeeze(I));
Shat_val    = predict(net_normalDrop,squeeze(I));

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

test_compression = 1;

S0          = squeeze(Data_temp.Sval);

opd_ind_drp = randperm(size(Data_temp.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-test_compression)));
I           = Data_temp.Ival;
I(:,:,:,opd_ind_drp,:,:) = 0;

%Shat_val    = predict(net_chDrop,squeeze(I));
%Shat_val    = predict(net_normalDrop,squeeze(I));
Shat_val    = predict(net_chDrop_p1,squeeze(I));

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
  
  Shat_nDrop_p5     = predict(net_nDrop_p5, squeeze(I));
  Shat_chDrop_p5    = predict(net_chDrop_p5,squeeze(I));
  Shat_chDrop_p1    = predict(net_chDrop_p1,squeeze(I));
  
  MSE_nDrop_p5(i,1) = mean((Shat_nDrop_p5 (:) - S0(:)).^2);
  MSE_chDrop_p5(i,1)= mean((Shat_chDrop_p5(:) - S0(:)).^2);
  MSE_chDrop_p1(i,1)= mean((Shat_chDrop_p1(:) - S0(:)).^2);
end

table(MSE_nDrop_p5,MSE_chDrop_p5,MSE_chDrop_p1)

imagesc(log10([MSE_nDrop_p5,MSE_chDrop_p5,MSE_chDrop_p1]));colorbar

%% Apply to real data
I           = (Data_temp.Itst - Data_temp.Itst(:,:,:,1,:,:))/100;

test_compression = 1;
opd_ind_drp = randperm(size(Data_temp.Ival,4));
opd_ind_drp = opd_ind_drp(1:round(length(opd_ind_drp)*(1-test_compression)));

I(:,:,:,opd_ind_drp,:,:) = 0;

Shat_nDrop_p5     = predict(net_nDrop_p5, squeeze(I));

plot(reshape(Shat_nDrop_p5,[128*128 142])')

plot(squeeze(Shat_nDrop_p5(97,67,:)));




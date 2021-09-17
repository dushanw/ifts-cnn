% 2021-05-26 by Dushan N. Wadduwage
% read ifts-data, and the spectral database

function Data = f_readData()

  %% read ifts data
%   crop_rect       = [321, 1170, 456, 252];
%   Data.Y_lsr488nm = single(subf_crop(subf_readInterfData('./488nm_500/'            ,499),crop_rect));
%   Data.Y_whiteLED = single(subf_crop(subf_readInterfData('./WhiteLED_500/'         ,499),crop_rect));
%   Data.Y_redLED   = single(subf_crop(subf_readInterfData('./RedLED_take2_forBeads/',499),crop_rect));
%   Data.Y_beads1A  = single(subf_crop(subf_readInterfData('./Beads_1A/'             ,499),crop_rect));
%   save('ifts_data.mat','Data')
  load ifts_data
  
  %% calibration
  lambda_lsr      = 488;                        % [nm]  
  [opd k]         = subf_calib1(Data.Y_lsr488nm,lambda_lsr);
                                                % [nm^-1], k = 1/lambda = wave number
  % manually compare and recorded based on WLQBF data
  cpi             = [500.6 522.0 549.8 621.1 670.7];
  oos             = [501.5 521.1 547.9 617.5 661.1];

  calib2_Trnsfrm  =fit(1./cpi',1./oos','poly2');
  k_calbed        = calib2_Trnsfrm(k);
  opd_tilt        = subf_calib_0opdPos(Data.Y_whiteLED,k);
  
  Data.opd        = opd;
  Data.opd_tilt   = opd_tilt;
  
  % correct opd-tilt on all datasets
  Data.Y_lsr488nm = subf_preProcI(Data.Y_lsr488nm,opd_tilt);
  Data.Y_whiteLED = subf_preProcI(Data.Y_whiteLED,opd_tilt);
  Data.Y_redLED   = subf_preProcI(Data.Y_redLED  ,opd_tilt);
  Data.Y_beads1A  = subf_preProcI(Data.Y_beads1A ,opd_tilt);
  
  %% read spectral database
  specBank.lambda = [1:ceil(max(1./k_calbed))]'; % [nm] full wavelength range
  
  file_listing    = dir('./spectra/Alexa Fluor/');  
  file_listing    = file_listing(3:end);        % remove '.' and '..' in mac
  specBank.ex     = zeros(length(specBank.lambda),length(file_listing));
  specBank.em     = zeros(length(specBank.lambda),length(file_listing));
  for i=1:length(file_listing)
    temp_var      = readmatrix([file_listing(i).folder '/' file_listing(i).name]);
    temp_lambda   = temp_var(:,1);
    
    specBank.names{i}           = file_listing(i).name;    
    specBank.ex(temp_lambda,i)  = temp_var(:,2);
    specBank.em(temp_lambda,i)  = temp_var(:,3);  
  end
  specBank.ex(specBank.ex(:)<0) = 0;
  specBank.em(specBank.em(:)<0) = 0;

  specBank_calbed.k             = k_calbed;
  specBank_calbed.lambda        = 1./k_calbed;
  specBank_calbed.ex            = specBank.ex(round(specBank_calbed.lambda),:);
  specBank_calbed.em            = specBank.em(round(specBank_calbed.lambda),:);
  
  Data.specBank                 = specBank;
  Data.specBank_calbed          = specBank_calbed;
    
end

function I        = subf_readInterfData(path,i_end)
    of = cd(path);
    
    if (i_end>99 & i_end<1000) 
        I_temp = imread(sprintf('img_%0.3d.tif',1));
        I = zeros(size(I_temp,1),size(I_temp,2),i_end+1);
        for i=0:i_end
            I(:,:,i+1)=imread(sprintf('img_%0.3d.tif',i));
            i
        end
    else % for i_end in thousends (mostly Raman data)
        I_temp = imread(sprintf('img_%0.4d.tif',1));
        I = zeros(size(I_temp,1),size(I_temp,2),i_end+1);
        for i=0:i_end
            I(:,:,i+1)=imread(sprintf('img_%0.4d.tif',i));
            i
        end
    end    
    cd(of);
    clc
end

function [I rect] = subf_crop(I,rect)
    if nargin==1
        Itemp = var(I,[],3);
        [Itemp rect] = imcrop(Itemp./max(Itemp(:)));
        rect = round(rect);
        % save('crop_rect.mat','rect');
    end
    I = I(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:);
    close all
end

function [opd f]  = subf_calib1(I_laser,lambda_laser)

  k_laser = 1/lambda_laser;% clarify this point should have 2pi or not?
  N = size(I_laser,3);
  S_laser = abs(fft(I_laser,[],3));
  [temp n] = max(S_laser(:,:,2:N/2),[],3);    
  dk = k_laser./n;
  dOpd = 1./(dk*N);

  opd = -dOpd*N/2:dOpd:dOpd*N/2-1;
  f_max = 1./(mean(dOpd(:))*2);
  f = linspace(0,f_max,N/2);
  % mean(dOpd(:))
end

function opd_tilt = subf_calib_0opdPos(I,k)
  I               = subf_smootheImages(I,10);
  S               = fftshift(fft(I,[],3),3);

  lambda          = 1./k;
  k_pb            = zeros(size(S,3),1);
  k_pb_inds       = [length(k_pb)/2+1 - find(lambda>600 & lambda<700) ...
                     length(k_pb)/2+1 + find(lambda>600 & lambda<700)];
  k_pb(k_pb_inds) = 1;

  S(:,:,k_pb==0)  = 0;
  I_bpf           = ifft(fftshift(S,3),[],3);  
  I_bpf           = I_bpf ./ var(I_bpf,[],3);
  
  x_ref = squeeze(I_bpf(1,1,:));  
  for i=1:size(I_bpf,1)      
    for j=1:size(I_bpf,2)        
      x_now       = squeeze(I_bpf(i,j,:));      
      D(i,j)      = finddelay(x_ref,x_now);
    end        
  end

  opd_tilt        = modefilt(D,[9 9]);
  % imagesc(opd_tilt);axis image;colorbar
end

function I = subf_smootheImages(I,s)
    filt = fspecial('gaussian',[s s], 0.5);
    
    for i=1:size(I,3)
        I(:,:,i) = imfilter(I(:,:,i), filt, 'replicate');
    end
end

function I_pp = subf_preProcI(I,opd_tilt)
      
  % Equation XX in the paper 
  % I_dc          = I(:,:,1);     % assuming that far OPD is measured at first
  % I             = I - I_dc;

  opd_idx       = -min(opd_tilt(:))+1:size(I,3)-max(opd_tilt(:));
    
  I_pp    = single(zeros([size(I,1) size(I,2) length(opd_idx)]));
  for i=1:size(I,1)
    for j=1:size(I,2)
      I_pp(i,j,:) = I(i,j,opd_idx + opd_tilt(i,j));
    end
  end
  
end





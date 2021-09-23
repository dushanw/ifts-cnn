% 2021-09-19 by Dushan N. Wadduwage
% main code to generate training data through the fwd_model. This code was adapted from the main_fwd.m

function [DataAll Data pram] = f_gen_allData_beads(pram)
  
  %% load-expData+spectraData
  load('./_data/ifts_data_pp.mat')              % of = cd('./_data/'); Data = f_readData; cd(of)
  load('./_PSFs/PSFs_06-Jun-2021 17:21:16.mat') % PSFs = f_simPSFs(pram);
  
  I_exp               = Data.Y_beads1A(64:64+pram.Ny-1,320:320+pram.Nx-1,:);
  opd_key_tilt        = Data.opd_tilt(120:120+50-1,320:320+128-1);

  pram.N_opd0         = length(Data.opd);  
  pram.N_k            = length(Data.specBank_calbed.k);
  %% set-data-structures
  opd_keys            = round(pram.N_opd0/2)-pram.N_opd/2+1:round(pram.N_opd0/2)+pram.N_opd/2;  % take-mid-opd-range
                                                
  %                                            dim-order
  %                                            1  2  3          4           5               6
  %                                            x  y  k          opd         spci            batch
  k         = reshape(Data.specBank_calbed.k ,[1, 1, pram.N_k,  1,          1,              1]);
  opd_keys  = reshape(opd_keys               ,[1, 1, 1,         pram.N_opd, 1,              1]);
  opd       = reshape(Data.opd(opd_keys)     ,[1, 1, 1,         pram.N_opd, 1,              1]);
  Spectra   = reshape(Data.specBank_calbed.em,[1, 1, pram.N_k,  1,          pram.N_spectra, 1]);

  I_exp     = reshape(I_exp(:,:,opd_keys),[pram.Ny pram.Nx 1 pram.N_opd]);

  %% siulate input and output to the inv model
  DataAll.Itr   = zeros(pram.Ny, pram.Nx, 1       , pram.N_opd, 1, pram.Nb, 'single');
  DataAll.Str   = zeros(pram.Ny, pram.Nx, pram.N_k, 1         , 1, pram.Nb, 'single');
  
  opd_tilt      = 0; % to account for the corrected OPD tilt
  t             = 1;
  for i=1:round(pram.Nb/pram.N_mb)
    
    %% simulate beads object
    pram.Nz     = pram.N_spci*pram.N_mb*3;

    %             fill-factor  *   volume             / volume-of-a-6um-bead
    N_beads     = round(0.3 * (pram.Ny*pram.Nx*pram.Nz) / ((4/3)*pi*(3/pram.dx)^3))+1;
    pram.dz     = 6;
    X0          = f_genobj_beads3D_6um(N_beads,pram);
    z_int       = squeeze(sum(sum(X0,1),2));
    z_idxidx    = find(z_int>max(z_int)/5);
    z_idx       = z_idxidx(round(linspace(1,length(z_idxidx),pram.N_spci*pram.N_mb))); 
    X0          = X0(:,:,z_idx);
    X0          = reshape(X0,[pram.Ny pram.Nx 1 1 pram.N_spci pram.N_mb]);
    % imagesc(imtile(squeeze(X0)));axis image

    %% simulate beads spectra
    S0          = Spectra(:,:,:,:,randi(pram.N_spectra,1,pram.N_spci),:);
    S0          = pram.countsFactor * S0;

    %% fwd-model
                                    %  1       2        3       4           5           6
                                    %  y       x        k       opd         spci        batch   
    [I XS_g1]   = f_fwd(X0,...      % [pram.Ny pram.Nx  1       1           pram.N_spci pram.N_mb]
                        S0,...      % [1       1        pram.Nk 1           pram.N_spci pram.N_mb]
                        k,...       % [1       1        pram.Nk 1           pram.N_spci pram.N_mb]
                        opd,...     % [1       1        1       pram.N_opd  1           1        ]
                        PSFs,...    % [pram.Ny pram.Nx  1       1           1           1        ]
                        opd_tilt,...
                        pram);
                      
    %% cat data
    DataAll.Itr(:,:,:,:,:,t:t+size(I,6)-1) = I;         %temp-for-a-later-look: I_n = I./I(:,:,:,1,:,:) - 1;
    DataAll.Str(:,:,:,:,:,t:t+size(I,6)-1) = XS_g1;
    
    t = t+size(I,6)
  end
  rand_inds     = randperm(size(DataAll.Itr,6));  
  DataAll.Itr   = DataAll.Itr(:,:,:,:,:,rand_inds);
  DataAll.Str   = DataAll.Str(:,:,:,:,:,rand_inds);
  
  DataAll.Ival  = DataAll.Itr(:,:,:,:,:,1:pram.N_mb);
  DataAll.Sval  = DataAll.Str(:,:,:,:,:,1:pram.N_mb);
  DataAll.Itr   = DataAll.Itr(:,:,:,:,:,pram.N_mb+1:end);
  DataAll.Str   = DataAll.Str(:,:,:,:,:,pram.N_mb+1:end);
  
  DataAll.Itst  = I_exp;                                %temp-for-a-later-look: I_exp_n   = I_exp./I_exp(:,:,:,1,:,:) - 1; 
                                                        
  % set-pram from data read 
  pram.N_opd0         = length(Data.opd);
  pram.N_opd          = length(opd);
  pram.k              = k;
  pram.lambda         = 2*pi./k;
  pram.opd            = opd;
  pram.N_k            = length(k);
  pram.N_spectra      = size(Data.specBank_calbed.em,2);
  
  
end


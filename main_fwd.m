% 2021-05-27 by Dushan N. Wadduwage

clc;clear all

pram      = f_pram_init();
load('./_data/ifts_data.mat')

resDir    = sprintf('./_results/%s/',date);
mkdir(resDir)

I_exp             = Data.Y_beads1A(120:120+50-1,360:360+50-1,:);
opd_key_tilt      = Data.opd_tilt(120:120+50-1,360:360+50-1);
% I_exp               = Data.Y_beads1A;
% opd_key_tilt        = Data.opd_tilt;
opd_tilt            = mean(diff(Data.opd))*opd_key_tilt;

pram.N_opd0         = length(Data.opd);
pram.Ny             = size(I_exp,1);
pram.Nx             = size(I_exp,2);

%PSFs = f_simPSFs(pram);
load('./_PSFs/PSFs_06-Jun-2021 17:21:16.mat')

% dim-order : [x y k opd species batch]
pram.useGPU         = 0;
pram.N_opd          = 64;
pram.N_mb           = 1;                      % #instances in a mini batch

pram.fringeContrast = 0.5;
pram.countsFactor   = 10;
pram.eta_opd        = 50;                     % opd noise [nm]
pram.N_k            = length(Data.specBank_calbed.k);
pram.N_spectra      = size(Data.specBank_calbed.em,2);
pram.N_spci         = 4;                      % every image has N_spci species selected at random out of N_spectra


%% no-undersampling
pram.N_opd          = pram.N_opd0;
opd_keys            = 1:pram.N_opd0;

%% undersampling
% opd_keys          = [1 sort(randi([-min(opd_key_tilt(:))+1 pram.N_opd0-max([0; opd_key_tilt(:)])],...
%                                   [1 pram.N_opd-1]))];

%% fwd
%                                            dim-order
%                                            1  2  3          4           5               6
%                                            x  y  k          opd         spci            batch   
k         = reshape(Data.specBank_calbed.k ,[1, 1, pram.N_k,  1,          1,              1]);
opd_keys  = reshape(opd_keys               ,[1, 1, 1,         pram.N_opd, 1,              1]);
opd       = reshape(Data.opd(opd_keys)     ,[1, 1, 1,         pram.N_opd, 1,              1]);
Spectra   = reshape(Data.specBank_calbed.em,[1, 1, pram.N_k,  1,          pram.N_spectra, 1]);

I_exp     = reshape(I_exp(:,:,opd_keys),[pram.Ny pram.Nx 1 pram.N_opd]);

% simulate beads
pram.Nz   = pram.N_spci*pram.N_mb*3;
%             fill-factor  *   volume             / volume-of-a-6um-bead
N_beads   = round(0.1 * (pram.Ny*pram.Nx*pram.Nz) / ((4/3)*pi*(3/pram.dx)^3))+1;
pram.dz   = 6;
X0        = f_genobj_beads3D_6um(N_beads,pram);
z_int     = squeeze(sum(sum(X0,1),2));
z_idxidx  = find(z_int>max(z_int)/2);
z_idx     = z_idxidx(round(linspace(1,length(z_idxidx),pram.N_spci*pram.N_mb)));
%X0       = X0(:,:,randi(size(X0,3),[1 pram.N_spci*pram.N_mb]));
X0        = X0(:,:,z_idx);
X0        = reshape(X0,[pram.Ny pram.Nx 1 1 pram.N_spci pram.N_mb]);
%imagesc(imtile(squeeze(X0)));axis image

S0        = Spectra(:,:,:,:,randi(pram.N_spectra,1,pram.N_spci),:);
S0        = pram.countsFactor * S0;
                              %  1       2        3       4           5           6
                              %  y       x        k       opd         spci        batch   
[I XS_g1] = f_fwd(X0,...      % [pram.Ny pram.Nx  1       1           pram.N_spci pram.N_mb]
                  S0,...      % [1       1        pram.Nk 1           pram.N_spci pram.N_mb]
                  k,...       % [1       1        pram.Nk 1           pram.N_spci pram.N_mb]
                  opd,...     % [1       1        1       pram.N_opd  1           1        ]
                  PSFs,...    % [pram.Ny pram.Nx  1       1           1           1        ]
                  opd_tilt,...
                  pram);

I_n       = I./I(:,:,:,1,:,:) - 1;
I_exp_n   = I_exp./I_exp(:,:,:,1,:,:) - 1; 

%% plot
for i=1:pram.N_mb
  [y1     x1    ] = find(mean(I(:,:,:,:,:,i),4)==max(max(mean(I(:,:,:,:,:,i),4))));
  [y1_exp x1_exp] = find(mean(I_exp,         4)==max(max(mean(I_exp,         4))));
  
  subplot(2,2,1);imagesc(mean(I_exp,4));axis image;axis off;colorbar
    hold on; plot(x1_exp,y1_exp,'X','MarkerSize',20);hold off
  subplot(2,2,2);imagesc(mean(I(:,:,:,:,:,i),4));axis image;axis off;colorbar
    hold on; plot(x1,y1,'X','MarkerSize',20);hold off
  subplot(2,2,3);plot(squeeze(I_exp_n(y1_exp,x1_exp,:)));%ylim([-1 1])
  subplot(2,2,4);plot(squeeze(I_n(y1,x1,:,:,:,i)));%ylim([-1 1])
  
  % exportgraphics(gcf,[resDir sprintf('fig_ifts_%d.png',i)],'Resolution',330)
  % saveas(gcf,[resDir sprintf('fig_ifts_%d.fig',i)])
end

%% preprocess for inverse model
I_pp      = f_preProc(I,...           % [pram.Ny pram.Nx  1       pram.N_opdKeys  1           1    ]
                      opd_keys,...    % [1       1        1       pram.N_opdKeys                   ]
                      opd_key_tilt,...% [pram.Ny pram.Nx]
                      pram);

%% inverse model















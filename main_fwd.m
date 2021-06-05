% 2021-05-27 by Dushan N. Wadduwage

clc;clear all

pram      = f_pram_init();
load('./_data/ifts_data.mat')
load('./_PSFs/PSFs04-Jan-2021 00_40_18.mat')
resDir    = sprintf('./_results/%s/',date);
mkdir(resDir)

I_exp               = Data.Y_beads1A(120:120+50-1,360:360+50-1,:);
opd_tilt            = mean(diff(Data.opd))*Data.opd_tilt(120:120+50-1,360:360+50-1);

pram.Ny             = size(I_exp,1);
pram.Nx             = size(I_exp,2);
pram.Nz             = 30;

% dim-order : [x y k opd species batch]

pram.fringeContrast = 0.5;
pram.countsFactor   = 10;
pram.eta_opd        = 50;                      % opd noise [nm] 
pram.N_mb           = 2;                       % #instances in a mini batch 
pram.N_k            = length(Data.specBank_calbed.k);
pram.N_opd          = length(Data.opd);
pram.N_spectra      = size(Data.specBank_calbed.em,2);
pram.N_spci         = 4;                       % every image has N_spci species selected at random out of N_spectra

%                                            dim-order
%                                            1  2  3          4           5               6
%                                            x  y  k          opd         spci            batch   
k         = reshape(Data.specBank_calbed.k ,[1, 1, pram.N_k,  1,          1,              1]);
opd       = reshape(Data.opd               ,[1, 1, 1,         pram.N_opd, 1,              1]);
Spectra   = reshape(Data.specBank_calbed.em,[1, 1, pram.N_k,  1,          pram.N_spectra, 1]);

I_exp     = reshape(I_exp,[pram.Ny pram.Nx 1 pram.N_opd]);

X0        = f_genobj_beads3D_6um(10,pram);
X0        = X0(:,:,randi(size(X0,3),[1 pram.N_spci*pram.N_mb]));
X0        = reshape(X0,[pram.Ny pram.Nx 1 1 pram.N_spci pram.N_mb]);

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

for i=1:pram.N_mb
  [y1     x1    ] = find(mean(I(:,:,:,:,:,i),4)==max(max(mean(I(:,:,:,:,:,i),4))));
  [y1_exp x1_exp] = find(mean(I_exp,         4)==max(max(mean(I_exp,         4))));
  
  subplot(2,2,1);imagesc(mean(I_exp,4));axis image;axis off;colorbar
    hold on; plot(x1_exp,y1_exp,'X','MarkerSize',20);hold off
  subplot(2,2,2);imagesc(mean(I(:,:,:,:,:,i),4));axis image;axis off;colorbar
    hold on; plot(x1,y1,'X','MarkerSize',20);hold off
  subplot(2,2,3);plot(squeeze(I_exp_n(y1_exp,x1_exp,:)));%ylim([-1 1])
  subplot(2,2,4);plot(squeeze(I_n(x1,y1,:,:,:,i)));%ylim([-1 1])
  
  exportgraphics(gcf,[resDir sprintf('fig_ifts_%d.png',i)],'Resolution',330)
end

%% temp <copy paste where needed when needed>
% load('temp_X0.mat'); % for the hardcoded y1,x1 and y2,x2
% y1        = 25;
% x1        = 20;
% y2        = 38;
% x2        = 45;
% y1_exp    = 38;
% x1_exp    = 33;

subplot(1,2,1);plot(squeeze(I_exp(y1_exp,x1_exp,:)));
subplot(1,2,2);plot(squeeze(I(x1(i),y1(i),:,:,:,i)));




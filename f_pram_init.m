% notes: 
%   The parameters: NA 1.0; Size of camera pixel on sample 330nm; exc/em wavelength: 800nm/590nm. [Cheng 2020-12-24]


function pram = f_pram_init()

  %% names
  pram.sim2dOr3d    = '2D';                               % {'2D','3D'}
  pram.mic_typ      = 'ifts';                             % {'ifts'}  
  pram.dataset      = 'beads';                            % {'minist',
                                                          %  'andrewCells_fociW3_63x_maxProj',
                                                          %  'andrewCells_dapi_20x_maxProj',
                                                          %  'beads'}
  pram.psf_typ      = 'sim_2d';                           % {'sim_2d','gaussian',...}
    
  %% data size parameters
  pram.Nx      = 256;
  pram.Ny      = 256;
  pram.Nz      = 1;
  pram.Nc      = 1;
  pram.Nt      = 250;
  pram.Nb      = 1e4;                                     %           number of batches (instances)
  pram.dz      = 1;
  pram.dx      = 0.2222;                                  % [um]      readme2 says its 100um/690px = 0.1449[um]
                                                          %           from 6um beads it's 6um/27px = 0.2222 [um]
    
  %% MIC and imaging parameters
  pram.lambda_ex  = 0.488;                                % [um]      excitation wavelength
  pram.lambda_em  = 0.515;                                % [um]      nominal emission wavelength mid(506,511,524)=515
  pram.NA         = 0.75;                                 % [AU]      numerical aperture of the objective. 
                                                          %           optica paper says 0.75 (Nikon, 40x, 0.75 NA, CFI Plan Fluor DLL)
  pram.nm         = 1;                                    % [AU]      refractive index. =1 becasue of air objective    
  
  %% camera parameters ??
    
  %% run environment parameters  
  pram.useGPU   = gpuDeviceCount>1 ;
  
end





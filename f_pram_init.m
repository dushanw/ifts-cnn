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
  pram.dx      = 0.2;
  pram.dz      = 1;
  
  %% MIC and imaging parameters
  pram.lambda_ex  = 0.800;                                % [um]      excitation wavelength
  pram.lambda_em  = 0.590;                                % [um]      emission wavelength {0.606 }
  pram.NA         = 1;                                    % [AU]      numerical aperture of the objective
    
  %% camera parameters ??
    
  %% run environment parameters  
  pram.useGPU   = gpuDeviceCount>1 ;
  
end





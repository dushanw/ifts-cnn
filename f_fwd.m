                                        %  1       2        3       4           5           6
                                        %  y       x        k       opd         spci        batch   
function [I XS_g1] = f_fwd( X0,...      % [pram.Ny pram.Nx  1       1           pram.N_spci pram.N_mb]
                            S0,...      % [1       1        pram.Nk 1           pram.N_spci pram.N_mb]
                            k,...       % [1       1        pram.Nk 1           pram.N_spci pram.N_mb]
                            opd,...     % [1       1        1       pram.N_opd  1           1        ]
                            PSFs,...    % [pram.Ny pram.Nx  1       1           1           1        ]
                            opd_tilt,...
                            pram)

  emPSF   = imresize(PSFs.emPSF,PSFs.pram.dx/pram.dx,'bilinear');

  if pram.useGPU ==1
    X0    = gpuArray(X0);
    S0    = gpuArray(S0);
    k     = gpuArray(k);
    opd   = gpuArray(opd);
    emPSF = gpuArray(emPSF);
  end
      
  X0_g1   = f_conv2nd(X0,emPSF,'same'); % image cube with spci dimention on
  XS_g1   = sum(X0_g1.*S0,5);           % image cube with k dimention on

  % Equation XX in the paper
  eta_opd = pram.eta_opd * (rand([pram.Ny pram.Nx 1 pram.N_opd  1 pram.N_mb])-0.5);
  I       = (1/8) * sum(XS_g1 .* (1 + pram.fringeContrast * cos(k .* (opd +...
                                                                      eta_opd + ...
                                                                      opd_tilt) ...
                                                               )...
                                 ),...
                        3);
  
  I       = poissrnd(I);  
end
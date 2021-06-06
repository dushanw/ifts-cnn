% 2021-06-06 by Dushan N. Wadduwage
% A wrapper to call the optical_PSF subtoolbox

function PSFs = f_simPSFs(pram)

  of = cd('_submodules/optical_PSF/');
  APSF_3D     = Efficient_PSF(pram.NA, pram.nm, pram.lambda_ex, pram.dx,pram.Nx-2,pram.Nx-2,2,200);
  PSF_3D      = abs(APSF_3D{1}).^2+abs(APSF_3D{2}).^2+abs(APSF_3D{3}).^2;
  exPSF       = PSF_3D(:,:,2).^2; % 2021-04-13 check with Peter if this dependence is correct. 
  exPSF       = exPSF/sum(exPSF(:));
  
  APSF_3D     = Efficient_PSF(pram.NA, pram.nm, pram.lambda_em, pram.dx,pram.Nx-2,pram.Nx-2,2,200);
  PSF_3D      = abs(APSF_3D{1}).^2+abs(APSF_3D{2}).^2+abs(APSF_3D{3}).^2;
  emPSF       = PSF_3D(:,:,2);
  emPSF       = emPSF/sum(emPSF(:));
  cd(of);
    
  %% save PSFs
  PSFs.exPSF  = exPSF;
  PSFs.emPSF  = emPSF;  
  PSFs.pram   = pram;
        
  save(['./_PSFs' datestr(datetime('now')) '.mat'],'PSFs','-v7.3'); % save sPSF
end
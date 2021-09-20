% 2021-09-20 by Dushan N. Wadduwage
% main code

%% model parameters
pram = f_pram_init();

%% generate training + testing data (contans the fwd model)
[Data pram] = f_gen_allData_beads(pram);

%% generate inv model
lgraph = f_gen_inv(pram);

%% train inv model

%% test inv model

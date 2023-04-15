clear;
clc;


data_path = 'new path';



% take Handwritten for an example
data_name = '3v_CCV';
fprintf('\ndata_name: %s', data_name);

%% pre-process, all algs do so
load([data_path, data_name, '.mat']);
        
k = length(unique(Y));
V = length(X);
X_bar = [];
for v = 1 : V    
    X_bar = [X_bar,X{v}];
    X{v} = mapstd(X{v},0,1)';
end
X_bar = mapstd(X_bar,0,1)';
 
%% para setting
m = k;
m_bar = k;
d_bar = size(X_bar, 1);
ii = 1;
for maxiter = 20
for lambda2 = [10]
   for tau = [1E-3]
      for lambda1 = 500
          fprintf('\nii = %d;', ii);
          tic;
          [UU, D, A, Z, Z_bar, Q, obj] = algo_qp(X, X_bar, Y, tau, lambda1, lambda2, d_bar, m_bar, m, maxiter); % X,Y,lambda,d,numanchor
          res = myNMIACCwithmean(UU,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
          time = toc;
          fprintf('\n ACC: %.4f NMI: %.4f PUR: %.4f FSC+-30-0+6*9/: %.4f PRE: %.4f REC: %.4f ARI: %.4f ENT: %.4f Time: %.4f \n',[res(1) res(2) res(3) res(4) res(5) res(6) res(7) res(8) time]);                  
          result(ii, :) = [res(1) res(2) res(3) res(4) res(5) res(6) res(7) res(8) time];
          ii = ii + 1;    
      end
   end
end
end 

ZZT = Z_bar' * Z_bar;
imagesc(ZZT)
colormap(gray(256))




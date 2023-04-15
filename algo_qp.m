function [UU, D, A, Z, Z_bar, Q, obj] = algo_qp(X, X_bar, Y, tau, lambda1, lambda2, d_bar, m_bar, m, maxiter)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.
% X      : n*di

%% initialize
numview = length(X);
numsample = size(Y, 1);

D = zeros(d_bar, m_bar);
Q = zeros(m, m_bar);
Z = zeros(m, numsample);
Z_bar = zeros(m_bar, numsample);
Z(:, 1 : m) = eye(m);
Z_bar(:, 1 : m_bar) = eye(m_bar);
A = zeros(m, m);
for i = 1:numview
   d(i) = size(X{i}, 1);
   W{i} = zeros(d(i), m);
end

opt.disp = 0;
flag = 1;
iter = 0;
%%
fprintf(' iter =');
while flag
    iter = iter + 1;
    fprintf(' %d', iter)
    
    %% optimize W
    parfor i = 1 : numview
        [U, ~, V] = svd(X{i} * Z' * A', 'econ');
        W{i} = U * V';
    end
    clear U V
    
    %% optimize A
    temp = zeros(m, m);
    for i = 1 : numview
        temp = temp + W{i}' * X{i} * Z';
    end
    [U, ~, V] = svd(temp, 'econ');
    A = U * V';
    clear U V temp
    

    %% optimize D
    [U, ~, V] = svd(X_bar * Z_bar', 'econ');
    D = U * V';
    clear U V
%     D = (X_bar * Z_bar') / (Z_bar * Z_bar' + 1e-4 * eye(m_bar));
    
    %% optimize Q
    %v  = sqrt(sum(Q .* Q, 1) + eps);
  %  vv  = diag(1 ./ (v));
  %  Q = (lambda1 * Z * Z_bar') / (lambda1 * Z_bar * Z_bar' + lambda2 * vv);
  %  clear v vv   
   Q = (lambda1 * Z * Z_bar') / (lambda1 * Z_bar * Z_bar' + lambda2 * eye(m_bar));
  %  clear v vv   
    
    
    %% optimize Z_bar
    Z_bar = (lambda1 * Q' * Q + (tau + lambda1) * eye(m_bar)) \ (tau * D' * X_bar + lambda1 * Q' * Z);
    
           
    %% optimize Z
    H = 2 * (numview + 2 * lambda1) * eye(m);
    H = (H + H') / 2;
    options = optimset( 'Algorithm','interior-point-convex','Display','off'); % Algorithm é»è®€äž? interior-point-convex
    for j = 1 : numsample
        ff = 0;
        for i = 1 : numview
          ff =  ff - 2 * X{i}(:, j)' * W{i} * A;
        end
        ff = ff - 2 * lambda1 * Z_bar(:, j)' * Q';
        Z(:, j) = quadprog(H, ff',[],[],ones(1, m),1,zeros(m, 1),ones(m, 1),[], options);
    end           


        
    %% calculate function value and check convergence
    term4 = norm(X_bar - D * Z_bar, 'fro') ^ 2;
    term1 = 0;
    for i = 1 : numview
       term1 = term1 + norm(X{i} - W{i} * A * Z, 'fro') ^ 2;
    end
    term2 = norm(Z - Q * Z_bar, 'fro') ^ 2;
    term3 = sum(sqrt(sum(Q .* Q, 1)));
    
    obj(iter) = numview \ term1 +  lambda1 * term2 + lambda2 * term3 + tau * term4;
    

    %if (iter>9) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter || obj(iter) < 1e-10)
    if iter>=maxiter
        [UU, ~, ~]=svd(Z', 'econ');
        flag = 0;
    end
end
         
         
    

% ---------------------------------------------------------------------
% Function: model = kspxy(X,Y,N,s)
% ---------------------------------------------------------------------
% Input:
% X, matrix of predictor variables.
% Y, vector of the response variable.
% N, number of objects to be selected to model set.
% s, width of the Gaussian kernel function.
% ---------------------------------------------------------------------
% Output:
% model, vector of objects selected to model set.
%----------------------------------------------------------------------
function [model,test] = kspxy(X,Y,N,s)
% Initializes the vector of minimum kernel distances.
kdminmax = zeros(1,N);
% Number of rows in X.
M = size(X,1); samples = 1:M;
% Auto-scales the Y matrix
for i=1:size(Y,2) % For each parameter in Y.
yi = Y(:,i);
Y(:,i) = (yi - mean(yi))/std(yi);
end

KD = zeros(M,M); % Initializes the matrix of X kernel distances.
KDy = zeros(M,M); % Initializes the matrix of Y kernel distances.

for i=1:M-1
xa = X(i,:);
ya = Y(i,:);
for j = i+1:M
xb = X(j,:);
yb = Y(j,:);
KD(i,j) = 2-2*exp(-(norm(xa-xb))^2/(2*s^2));
KDy(i,j)=2-2*exp(-(norm(ya-yb))^2/(2*s^2));
end
end

KDmax = max(max(KD)); KDymax = max(max(KDy));
D = KD/KDmax + KDy/KDymax;  % Combines the kernel distances in X and Y.
% D: Upper Triangular Matrix.
% KD(i,j) = Kernel distance between objects i and j (j > i).
% maxD = Row vector containing the largest element of each column in D.
% index_row(n) = Index of the row with the largest element in the n-th column.
[maxD,index_row] = max(D);
% index_column = column corresponding to the largest element in matrix D.
[dummy,index_column] = max(maxD); model(1) = index_row(index_column);
model(2) = index_column;
Dminmax(2) = D(model(1),model(2)); 
for i = 3:N
% This routine determines the kernel distances between each sample
% still available for selection and each of the samples already selected.
% pool = Samples still available for selection.
pool = setdiff(samples,model);
% Initializes the vector of minimum kernel distances between each.
% sample in pool and the samples already selected.
kdmin = zeros(1,M-i+1);
% For each sample xa still available for selection.
for j = 1:(M-i+1)
% indexa = index of the j-th sample in pool (still available for selection).
indexa = pool(j);
% Initializes the vector of kernel distances between the j-th
% sample in pool and the samples already selected.
kd = zeros(1,i-1);
% The kernel distance with respect to each sample already selected.
% is analyzed
for k = 1:(i-1)
% indexb = index of the k-th sample already selected.
indexb = model(k);
if indexa < indexb
kd(k) = D(indexa,indexb);
else
kd(k) = D(indexb,indexa);
end
end
kdmin(j) = min(kd);
end
% The selected sample corresponds to the largest kdmin.
[kdminmax(i),index] = max(kdmin);
model(i) = pool(index);
end
if nargout==2
    test=samples;
    test(model)=[];
end
%----------------------------------------------------------------------


%dvenugopalarao%

clc
clear 
close all

rng default

[data,txt,raw]=xlsread('N2-Data.xlsx',1);
       
x=[data(:,1:4) data(:,5)];
y=data(:,7);

Ns=floor(0.8*length(data));
[xtrain,xtest,ytrain,ytest]=train_test_data(x,y,'HS',Ns,0);

% 5-fold cross validation to find optimum number of neighbors
ns=floor(length(xtrain)/5);

% x1=xtrain(1:ns,:);y1=ytrain(1:ns);
% x2=xtrain(ns+1:2*ns,:);y2=ytrain(ns+1:2*ns);
% x3=xtrain(2*ns+1:3*ns,:);y3=ytrain(2*ns+1:3*ns);
% x4=xtrain(3*ns+1:4*ns,:);y4=ytrain(3*ns+1:4*ns);
% x5=xtrain(4*ns+1:5*ns,:);y5=ytrain(4*ns+1:5*ns);

metric = 'euclidean';
weights = 'distance'; %{'uniform', 'distance'};

Neighbors=[1:1:15]; 
for i=1:length(Neighbors)
    a=1;b=ns;
    for k=1:5
        
        b=k*ns;
        testx=xtrain(a:b,:);testy=ytrain(a:b);
        
        if k==1
            trainx=[xtrain(ns+1:end,:)];trainy=[ytrain(ns+1:end)];
        else
            trainx=[xtrain(1:(k-1)*ns,:);xtrain(k*ns+1:end,:)];trainy=[ytrain(1:(k-1)*ns);ytrain(k*ns+1:end)];
        end
            
        [trainx,mux,sigmax] = zscore(trainx);
        [trainy,muy,sigmay] = zscore(trainy);
        
        xnew=(testx-mux)./sigmax; % test dataset
        
        mdl = kNNeighborsRegressor(Neighbors(i),metric,weights);
        mdl = mdl.fit(trainx,trainy');
        Ypred = mdl.predict(xnew);
        Ypred=Ypred*sigmay+muy;
        [Ypred' testy]
        mse(k)=mean((testy-Ypred').^2);
        a=b+1;
        clear mdl
        
    end
    MSE(i)=mean(mse);
    clear mse
end

plot([1:15],MSE,'LineWidth',1.5)
xlabel('Nearest neighbors, k')
ylabel('Average MSE')

[~,idx]=min(MSE);

% knn regression with optimum neighbors

[xtrain,mux,sigmax] = zscore(xtrain);
[ytrain,muy,sigmay] = zscore(ytrain);
        
xnew=(xtest-mux)./sigmax; % test dataset
% xnew=xtrain;

mdl = kNNeighborsRegressor(Neighbors(idx),metric,weights);
mdl = mdl.fit(xtrain,ytrain');
ypred = mdl.predict(xnew);

ypred=ypred'*sigmay+muy;
 
% ytrain=ytrain*sigmay+muy;
% ytest=ytrain;

R=(corr(ytest,ypred)^2);
fprintf('R^2= %4.4f \n',R)

sse=sum((ypred-ytest).^2);
sst=sum((ytest-mean(ytest)).^2);
R2=1-(sse/sst);
% disp(R2)

AARD=100*mean(abs((ypred-ytest)./ytest));
fprintf('AARD= %4.4f \n',AARD)

RMSE=sqrt(mean((ypred-ytest).^2));
fprintf('RMSE= %4.4f \n',RMSE)

figure
plot(ytest)
hold on
plot(ypred)
legend('data','prediction')

% average percent relative error
E=((ytest-ypred)./ytest)*100;
Er=mean(E);
fprintf('Er= %4.4f \n',Er)

Ea=mean(abs(E));
fprintf('Ea= %4.4f \n',Ea)





% Based on W. D. Penny, "A Variational Bayesian Framework for d-dimensional Graphical
% Models"

clear all;
close all;

% dims = [2 4 8 16 32 64];
% for nDims=1:length(dims)
%     d = dims(nDims);
% d=100;
% parameters(1).mu = zeros(1,d);         % H01
% parameters(1).cov = eye(d);
% parameters(2).mu = 3*ones(1,d);        % H02
% parameters(2).cov = eye(d);
% bags = MILgenerateSetOneH1OneH0(100,10,parameters);

% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\msrcorid_Mil_Data');
% for nBag=1:length(dsMilHogPcaStruct)
%     instThisBag = size(dsMilHogPcaStruct(nBag).data,1);
%     dsMilHogPcaStruct(nBag).label = bagLabels(nBag)*ones(instThisBag,1);
%     dsMilHogPcaStruct(nBag).bagNum = nBag*ones(instThisBag,1);
% end
% bags = concatStructs(1,dsMilHogPcaStruct);

load('/Volumes/My Passport/duke/research/matlabStuffData/Research/milDatasets/bags200TwoH1TwoH0R2_5.mat');
% load('C:\Users\manandhar\research\matlabStuff\Research\bnpMilSynDataExp\synDataBags200TwoH1TwoH0XorR1d200c0_5\datasetsTrain\bagsSet1.mat');bags=bagsTrain;
% load('C:\Users\manandhar\research\matlabStuffData\Research\milDatasets\bags200TwoH1TwoH0R2_5');
% load('bags2004H1Gauss1H0Gauss');
% load('bags200TwoH1TwoH0R3');
% load bags100TwoGaussMean0033CovI2H1PerPosBag;

% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\musk1Norm');

% load('C:\Users\manandhar\research\matlabStuffData\Research\milDatasets\EHD_HMDS');
% bags = downSampleBags(bags,10); % make around 1000 H1 and 1000 H0

% % Preproc Xtrain
% ds = prtDataSetClass(bags.data,bags.label);
% objZmuvPca = prtPreProcZmuv+prtPreProcPca('nComponents',6);
% objZmuvPca = objZmuvPca.train(ds);
% dsPreProc = objZmuvPca.run(ds);
% bags.data = dsPreProc.data;
% % free memory
% clear ds objZmuvPca dsPreProc;
% bags.data = zscore(bags.data);
% dataset = prtDataSetClass(zscore(bags.data),bags.label);
% pca = prtPreProcPca;
% pca.nComponents = 10;
% pca = pca.train(dataset); % default 3 PC
% datasetNew = pca.run(dataset);
% bags.data = datasetNew.data; % [totalTimeSamples * default 3 PC]
% nPc = 20; d = nPc;
% bags.data = myPrinCompFun(bags.data,nPc);

% profile on;
myOptions.delay             = 1; % tau = 10 Don't down-weight early iterations quickly
myOptions.forgettingRate    = .55; % kappa = {.5,.6,...,1.0} = .55 % Don't forget early iterations quickly
myOptions.NBatches          = 20;
myOptions.plot = false;
myOptions.featsUncorrelated = false;
myOptions.rndInit = true;
myOptions.maxIteration = 500;
myOptions.pruneClusters = true;
myOptions.pruneThreshold = 0.1;
[posteriors,nfeTerms] = npbMilMvn(bags,10,10,myOptions);
% profile report;
% profile off;
nfeTermsCat = concatStructs(1,nfeTerms);

% figure(1);subplot(2,1,1);stem(posteriors(1).pi);subplot(2,1,2);stem(posteriors(2).pi);

% figure(2);
% subplot(2,4,1);
% plot(nfeTermsCat.nfe(:,1),'.');
% xlabel('iteration');
% ylabel('NFE');
% title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
% 
% subplot(2,4,2);
% plot(nfeTermsCat.Lav,'.');
% xlabel('iteration','FontWeight','b');
% ylabel('Lav','FontWeight','b');
% % title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
% 
% subplot(2,4,3);
% plot(nfeTermsCat.KLZeta,'.');
% xlabel('iteration','FontWeight','b');
% ylabel('KL(\zeta)','FontWeight','b');
% % title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
% 
% subplot(2,4,4);
% plot(nfeTermsCat.KLz,'.');
% xlabel('iteration','FontWeight','b');
% ylabel('KL(z)','FontWeight','b');
% % title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
% 
% subplot(2,4,5);
% plot(nfeTermsCat.KLeta,'.');
% xlabel('iteration','FontWeight','b');
% ylabel('KL(\eta)','FontWeight','b');
% % title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
% 
% subplot(2,4,6);
% plot(nfeTermsCat.KLv,'.');
% xlabel('iteration','FontWeight','b');
% ylabel('KL(v)','FontWeight','b');
% % title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
% 
% subplot(2,4,7);
% plot(nfeTermsCat.KLMuAndGamma,'.');
% xlabel('iteration','FontWeight','b');
% ylabel('KL(\mu,\Gamma)','FontWeight','b');
% % title(['dimension=',num2str(d),', KH1=10, KH0=10']);
% grid on;
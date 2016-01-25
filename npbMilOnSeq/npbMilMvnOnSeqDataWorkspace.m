clear all;
close all;

% Landmine Syn Mvn
pathToSynMvnNpbmil = 'C:\Users\manandhar\research\matlabStuff\Research\npbMil\npbMilOnNonSeq\hmmMilMvnR1CovIStates4SameObsModDiffTrans1';
pathToSynMvnFeats = 'C:\Users\manandhar\research\matlabStuff\Research\hmm\datasets\hmmMilMvnR1CovIStates4SameObsModDiffTrans1\Bags200I10H1OneTimeSamples15';
% load(fullfile(pathToSynMvnFeats,'Bags200I10H1OneTimeSamples15.mat'));
% 
% Nbags = length(unique(bags.bagNum));
% NDim = size(bags.data,2);
% % Assuming equal number of NWords per instance
% timeSamplesPerInstance = length(bags.data(bags.instanceNumber==1,:));
% % Assuming equal number of instances per bag
% instancesPerBag = length(unique(bags.instanceIndex(bags.bagNum==1,:)));
% 
% bagsNonSeq.data = reshape(bags.data',NDim*timeSamplesPerInstance,instancesPerBag*Nbags)';
% bagsNonSeq.bagNum = bags.bagNum(unique(bags.instanceIndex),:);
% bagsNonSeq.label = bags.label(unique(bags.instanceIndex),:);
% 
% clear bags;
% bags = bagsNonSeq;
% clear bagsNonSeq;
% save(fullfile(pathToSynMvnNpbmil,'Bags200I10H1OneTimeSamples15NonSeq'),'bags');

ds = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance(fullfile(pathToSynMvnNpbmil,'Bags200I10H1OneTimeSamples15NonSeq'));
load(fullfile(pathToSynMvnFeats,'10xValKeys1.mat'));

vbdp = prtClassMilVbDpGmm;
vbdp.K0 = 4;
vbdp.K1 = 4;
vbdp.rndInit = 1;
vbdp.plot = 1;
vbdp.convergenceThreshold = 1e-5;
[yOutVbdp,~,xValKeys] = kfolds(vbdp,ds,10);

[Pfa,Pd,~,aucVB] = prtScoreRoc(yOutVbdp);
figure;
plot(Pfa,Pd,'LineWidth',2);
title('ROC');
beep
% save(fullfile(pathToSynMvnNpbmil,'Pfa10xValNpbMil'),'Pfa');
% save(fullfile(pathToSynMvnNpbmil,'Pd10xValNpbMil'),'Pd');

%%
clear all;
close all;

% Landmine HOG
pathToLandHogNpbmil = 'C:\Users\manandhar\research\matlabStuff\Research\npbMil\npbMilOnNonSeq\2012_03_STMRbig_dsF1_HMMHog';
pathToLandHogFeats = 'C:\Users\manandhar\research\matlabStuff\Research\hmm\datasets\2012_03_STMRbig_dsF1_HMMHog\BagsDs5InstancesDs2';
% load(fullfile(pathToLandHogFeats,'STMRbig_dsF1_HMMHogBagsDs5InstancesDs2.mat'));
% 
% Nbags = length(unique(bags.bagNum));
% NDim = size(bags.data,2);
% % Assuming equal number of NWords per instance
% timeSamplesPerInstance = size(bags.data(bags.instanceNumber==1,:),1);
% % Assuming equal number of instances per bag
% instancesPerBag = length(unique(bags.instanceIndex(bags.bagNum==1,:)));
% 
% bagsNonSeq.data = reshape(bags.data',NDim*timeSamplesPerInstance,instancesPerBag*Nbags)';
% bagsNonSeq.bagNum = bags.bagNum(unique(bags.instanceIndex),:);
% bagsNonSeq.label = bags.label(unique(bags.instanceIndex),:);
% 
% dataset = prtDataSetClass(zscore(bagsNonSeq.data),bagsNonSeq.label);
% pca = prtPreProcPca;
% pca.nComponents = 36; % 90%
% pca = pca.train(dataset); % default 3 PC
% datasetNew = pca.run(dataset);
% bagsNonSeq.data = datasetNew.data;
% 
% clear bags;
% bags = bagsNonSeq;
% clear bagsNonSeq;
% save(fullfile(pathToLandDiscNpbmil,'STMRbig_dsF1_HMMHogBagsDs5InstancesDs2NonSeq'),'bags');

ds = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance(fullfile(pathToLandHogNpbmil,'STMRbig_dsF1_HMMHogBagsDs5InstancesDs2NonSeq'));
load(fullfile(pathToLandHogFeats,'10xValKeys.mat'));
vbdp = prtClassMilVbDpGmm;
vbdp.K0 = 10;
vbdp.K1 = 10;
vbdp.rndInit = 1;
vbdp.plot = 1;
vbdp.convergenceThreshold = 1e-5;
% [yOutVbdp,~,xValKeys] = kfolds(vbdp,ds,10);
yOutVbdp = vbdp.crossValidate(ds,crossValKeys);

[Pfa,Pd,~,aucVB] = prtScoreRoc(yOutVbdp);
figure;
plot(Pfa,Pd,'LineWidth',2);
title('ROC');
beep
% save(fullfile(pathToLandHogNpbmil,'Pfa10xValNpbMil'),'Pfa');
% save(fullfile(pathToLandHogNpbmil,'Pd10xValNpbMil'),'Pd');

%%
% myTrainFun          = @classifyVbDpMnMilLlrtTrain;
% myTestFun           = @classifyVbDpMnMilLlrtTest;
% myOptions.totalH0Clusters = 4;
% myOptions.totalH1Clusters = 4;
% myOptions.convergenceThreshold = 1e-3;
% %     myOptions.maxIteration = 250;
% %     myOptions.featsUncorrelated = 1;
% myOptions.rndInit = 1;
% numOfFolds          = 10;
% 
% % load('C:\Users\manandhar\research\matlabStuff\Research\bnpMil\savedVariables\trec1\roc10xTrec1Gt19RmFeat36NpbMilRndInit\10xValKeys');
% % myOptions.crossValKeys = crossValKeys;
% 
% %     profile on;
% tic;
% [yOut, xValTrainedClassifier] = nValMil(bagsNonSeq,numOfFolds,myTrainFun,myTestFun,myOptions); % leave-one-bag-out xVal
% timeForExecution = toc;    
% %     profile report;
% %     profile off; 
% 
% % Bag labels, only one per bag
% bagNums = unique(bags.bagNum);
% totalBags = length(bagNums);
% for nBag=1:totalBags
%     bagLabels(nBag,1) = unique(bags.label(bags.bagNum==nBag,:));
% end
% 
% % Percent Correct
% [pc, thresh]  = findPercentCorrect(bagLabels,yOut);
% 
% % ROC
% [Pfa,Pd,~,auc] = prtScoreRoc(yOut,bagLabels);auc
% figure(4);
% handle1 = plotRocFun(Pfa,Pd,'color','b','LineWidth',2);
    
clear all;
close all;

% Landmine Disc Syn Feats
pathToSynMnNpbmil = 'C:\Users\manandhar\research\matlabStuff\Research\npbMil\npbMilOnNonSeq\hmmMilDiscreteWords16States4H14H0SameObsModDiffTrans7';
pathToSynMnFeats = 'C:\Users\manandhar\research\matlabStuff\Research\hmm\datasets\hmmMilDiscreteWords16States4H14H0SameObsModDiffTrans7\Bags400I10H1OneNwords1TimeSamples15';
% load(fullfile(pathToSynMnFeats,'Bags400I10H1OneNwords1TimeSamples15.mat'));
% 
% % Assuming equal number of NWords per instance
% NWords = size(bags.data(bags.instanceNumber==1,:),2);
% feats = nan(length(unique(bags.instanceNumber)),NWords);
% for nWord=1:NWords
%     feats(:,nWord) = accumarray(bags.instanceNumber,bags.data(:,nWord));
% end
% bagsNonSeq.data = feats;
% bagsNonSeq.bagNum = bags.bagNum(unique(bags.instanceIndex),:);
% bagsNonSeq.label = bags.label(unique(bags.instanceIndex),:);
% bagsNonSeq.NWords = sum(bagsNonSeq.data,2);
% 
% clear bags;
% bags = bagsNonSeq;
% clear bagsNonSeq;
% % save(fullfile(pathToSynMnNpbmil,'Bags400I10H1OneNwords1TimeSamples15'),'bags');

ds = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance(fullfile(pathToSynMnNpbmil,'Bags400I10H1OneNwords1TimeSamples15NonSeq'));
load(fullfile(pathToSynMnFeats,'10xValKeys1.mat'));

vbdp = prtClassMilVbDpMn;
vbdp.K0 = 4;
vbdp.K1 = 4;
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
% save(fullfile(pathToSynMnNpbmil,'Pfa10xValNpbMil'),'Pfa');
% save(fullfile(pathToSynMnNpbmil,'Pd10xValNpbMil'),'Pd');

%%
clear all;
close all;

% % Landmine Disc Feats
% pathToLandDiscNpbmil = 'C:\Users\manandhar\research\matlabStuff\Research\npbMil\npbMilOnNonSeq\2012_03_STMRbig_dsF1_HMMObsSamp';
% pathToLandDiscFeats = 'C:\Users\manandhar\research\matlabStuff\Research\hmm\datasets\2012_03_STMRbig_dsF1_HMMObsSamp\BagsDs5InstancesDs5';
% load(fullfile(pathToLandDiscFeats,'STMRbig_dsF1_HMMObsSampBagsDs5InstancesDs5.mat'));
% bags.data = bsxfun(@eq,1:16,bags.data);

% Assuming equal number of NWords per instance
NWords = size(bags.data(bags.instanceNumber==1,:),2);
feats = nan(length(unique(bags.instanceNumber)),NWords);
for nWord=1:NWords
    feats(:,nWord) = accumarray(bags.instanceNumber,bags.data(:,nWord));
end
bagsNonSeq.data = feats;
bagsNonSeq.bagNum = bags.bagNum(unique(bags.instanceIndex),:);
bagsNonSeq.label = bags.label(unique(bags.instanceIndex),:);
bagsNonSeq.NWords = sum(bagsNonSeq.data,2);

clear bags;
bags = bagsNonSeq;
clear bagsNonSeq;
% Landmine Disc Feats
% save(fullfile(pathToLandDiscNpbmil,'STMRbig_dsF1_HMMObsSampBagsDs5InstancesDs5NonSeq'),'bags');

% Landmine Disc Feats
% ds = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance(fullfile(pathToLandDiscNpbmil,'STMRbig_dsF1_HMMObsSampBagsDs5InstancesDs5NonSeq'));
% load(fullfile(pathToLandDiscFeats,'10xValKeys.mat'));

vbdp = prtClassMilVbDpMn;
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
% Landmine Disc Feats
% save(fullfile(pathToLandDiscNpbmil,'Pfa10xValNpbMil'),'Pfa');
% save(fullfile(pathToLandDiscNpbmil,'Pd10xValNpbMil'),'Pd');

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
    
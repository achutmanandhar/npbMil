
clear all
close all

% load msrcorid_Mil_Data.mat dsMilHogFullStruct dsMilHogPcaStruct bagLabels
% dsMIl = prtDataSetClassMultipleInstance(dsMilHogPcaStruct,bagLabels(:));
% dsMIL = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\musk1Norm.mat');
% dsMIL = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance('C:\Users\manandhar\research\matlabStuff\Research\bnpMilSynDataExp\synDataBags200TwoH1TwoH0XorR1d25c0_3\datasetsTrain\bagsSet1.mat');
dsMIL = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\bags200TwoH1TwoH0R2_5.mat');

vbdp = prtClassMilVbDpGmm;
vbdp.rndInit = 0;
vbdp.featsUncorrelated = 0;
vbdp.convergenceThreshold = 1e-12;
vbdp.plot = 1;
vbdp.K0 = 10;
vbdp.K1 = 10;
[yOutVbdp,~,xValKeys] = kfolds(vbdp,dsMIL,3);
% yOutVbdp = kfolds(vbdp,dsMilHog,3);

[pfVB,pdVB,~,aucVB] = prtScoreRoc(yOutVbdp);
figure;
plot(pfVB,pdVB,'LineWidth',2);
title('ROC 3 Fold Musk1');
% title('ROC 3 Fold 5-PC msrcorid; non-informative prior');
% title('ROC 3 Fold 5-PC msrcorid; strong prior');
beep

%%

% Pfa = pfVB;
% Pd = pdVB;
% auc = aucVB;
% save('C:\Users\manandhar\research\matlabStuff\Research\npbMil\PhiInitExp\msrcorid\PeteCodeModifiedRunActionButPeteInit\PfaIncest','Pfa');
% save('C:\Users\manandhar\research\matlabStuff\Research\npbMil\PhiInitExp\msrcorid\PeteCodeModifiedRunActionButPeteInit\PdIncest','Pd');
% save('C:\Users\manandhar\research\matlabStuff\Research\npbMil\PhiInitExp\msrcorid\PeteCodeModifiedRunActionButPeteInit\aucIncest','auc');
% % 
% load('C:\Users\manandhar\research\matlabStuff\Research\npbMil\PhiInitExp\msrcorid\PhiVb01\xValKeys');
% clear vbdp
% vbdp = prtClassMilVbDpGmm;
% [yOutVbdp] = vbdp.crossValidate(dsMIl,xValKeys);
% % yOutVbdp = kfolds(vbdp,dsMilHog,3);
% 
% [pfVB,pdVB,~,aucVB] = prtScoreRoc(yOutVbdp);
% figure;
% plot(pfVB,pdVB,'LineWidth',2);
% % title('ROC 3 Fold 5-PC msrcorid; non-informative prior');
% title('ROC 3 Fold 5-PC msrcorid; strong prior');
% beep
 
%%
% vbdp = prtClassMilVbDpGmmAchut;
% vbdp.rndInit = 0;
% vbdp.featsUncorrelated = 0;
% vbdp.plot = 1;
% vbdp = vbdp.train(dsMIl);
% yOutVbdp = vbdp.run(dsMIl);
% 
% [pfVB,pdVB,~,aucVB] = prtScoreRoc(yOutVbdp);
% figure;
% plot(pfVB,pdVB,'LineWidth',2);
% % title('ROC 3 Fold 5-PC msrcorid; non-informative prior');
% title('ROC 3 Fold 5-PC msrcorid; strong prior');
% beep

%%
ds = prtUtilManandaharMilBagStruct2prtDataSetClassMultipleInstance('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\trec1NonZeroFeatures.mat');
vbdp = prtClassMilVbDpMn;
vbdp.rndInit = 1;
vbdp.plot = 1;
% vbdp.convergenceThreshold = 1e-5;
[yOutVbdp,~,xValKeys] = kfolds(vbdp,ds,10);

[pfVB,pdVB,~,aucVB] = prtScoreRoc(yOutVbdp);
figure;
plot(pfVB,pdVB,'LineWidth',2);
title('ROC 3 Fold trec1');
beep
clear all
close all

load msrcorid_Mil_Data.mat dsMilHogFullStruct dsMilHogPcaStruct bagLabels
dsMIl = prtDataSetClassMultipleInstance(dsMilHogPcaStruct,bagLabels(:));

vbdp = prtClassMilVbDpGmmAchut;
vbdp.rndInit = 0;
vbdp.featsUncorrelated = 0;
vbdp.plot = 1;
[yOutVbdp,~,xValKeys] = kfolds(vbdp,dsMIl,3);
% yOutVbdp = kfolds(vbdp,dsMilHog,3);

[pfVB,pdVB,~,aucVB] = prtScoreRoc(yOutVbdp);
figure;
plot(pfVB,pdVB,'LineWidth',2);
title('ROC 3 Fold 5-PC msrcorid; non-informative prior');
% title('ROC 3 Fold 5-PC msrcorid; strong prior');
beep
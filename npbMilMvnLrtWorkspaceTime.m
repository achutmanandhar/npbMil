clear all;
close all;

addpath('C:\Users\manandhar\research\matlabStuff\Research\npbMil');
addpath('C:\Users\manandhar\research\matlabStuffData\Research\milBenchmarkTimeMissing');


dataSetName(1).name = 'musk1Pc20'; % poly
dataSetName(2).name = 'musk2Pc20'; % poly
dataSetName(3).name = 'elephantPc40'; % rbf
dataSetName(4).name = 'tigerPc40'; % lin
dataSetName(5).name = 'foxPc40'; % rbf
dataSetName(8).name = 'ehdDs10Pc6'; % rbf

dataFileName(1).name = 'musk1Pc20';
dataFileName(2).name = 'musk2Pc20';
dataFileName(3).name = 'elephantNonZeroFeatsPc40';
dataFileName(4).name = 'tigerNonZeroFeatsPc40';
dataFileName(5).name = 'foxNonZeroFeatsPc40';
dataFileName(8).name = 'ehdDs10Pc6';

algName = 'vbMil';

for nDataSet=8%1:length(dataSetName)

    pathName = 'C:\Users\manandhar\research\matlabStuffData\Research\milBenchmarkTimeMissing\nPC\';
    load([pathName,dataSetName(nDataSet).name,'\',dataFileName(nDataSet).name]);

    % Bag labels, only one per bag
    bagNums = unique(bags.bagNum);
    totalBags = length(bagNums);
    for nBag=1:totalBags
        myBagLabels(nBag,1) = unique(bags.label(bags.bagNum==bagNums(nBag),:));
    end

    for nIteration=2:10
        
        fprintf('\n Experiment Iteration %d...\n',nIteration);

        myTrainFun          = @npbMilMvnTrain;
        myTestFun           = @npbMilMvnTest;
        myOptions.totalH0Clusters = 20;
        myOptions.totalH1Clusters = 20;
        myOptions.plot = 0;
        myOptions.featsUncorrelated = 0;
        myOptions.rndInit = 1;
        numOfFolds          = 10;

        load([pathName,dataSetName(nDataSet).name,'\10xValKeys',num2str(nIteration)]);
        myOptions.crossValKeys = crossValKeys;

        [yOut,~,~,timeTrainFolds, timeTestFolds] = nValMilTimePP1(bags,numOfFolds,myTrainFun,myTestFun,myOptions); % leave-one-bag-out xVal 
       
        timeTrain = sum(timeTrainFolds)
        timeTest = sum(timeTestFolds)
        timeTrainVec(nIteration) = timeTrain;
        timeTestVec(nIteration) = timeTest;

        % Percent Correct
        [pc, thresh]  = findPercentCorrect(myBagLabels,yOut);
        pcVec(nIteration)=pc;

        % ROC
        [Pfa,Pd,~,auc] = prtScoreRoc(yOut,myBagLabels);

        figure(5);
        handle1 = plotRocFun(Pfa,Pd,'color','b','LineWidth',2);
        title(['10xVal ',dataSetName(nDataSet).name,' VBMIL',],'FontSize',10);  
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\timeTrain',num2str(nIteration)],'timeTrain');
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\timeTest',num2str(nIteration)],'timeTest');   
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\Pd',num2str(nIteration)],'Pd');
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\Pfa',num2str(nIteration)],'Pfa');
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\auc',num2str(nIteration)],'auc');
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\pc',num2str(nIteration)],'pc');
        save([pathName,dataSetName(nDataSet).name,'\',algName,'\thresh',num2str(nIteration)],'thresh');
        hold on;    
    end
    hold off;
    beep;
    pcVec
    timeTrainVec
    timeTestVec

end

% folderName1 = 'bnpMil';
% dataSetName = 'musk2';
% folderName2 = 'roc10xMusk2';
% folderName3 = 'BnpMilFeatsUncorrRndInit';

% folderName1 = 'bnpMil';
% dataSetName = 'elephant';
% folderName2 = 'roc10xElephantNonSparseSumFeatsNot0';
% folderName3 = 'NpbMilVbInit';

% folderName1 = 'bnpMil';
% dataSetName = 'msrcorid';
% % folderName2 = 'roc3xDsMilHog';
% % folderName2 = 'roc3xDsMilHogPc5';
% folderName2 = 'roc10xDsMilHogPc5';
% % folderName3 = 'NpbMilRndInit';
% folderName3 = 'NpbMilVbInit';
% % folderName3 = 'NpbMilFeatsUncorrRndInit';
% % folderName3 = 'NpbMilFeatsUncorrVbInit';
% for nIteration=1:10
%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\% savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\time',num2str(nIteration)]);
%     timeVec(nIteration) = timeForExecution;
% %     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\% savedVariables\',dataSetName,'\',...
% %         folderName2,folderName3,'\auc',num2str(nIteration)]);
% %     aucVec(nIteration) = auc;
%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\% savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\pc',num2str(nIteration)]);
%     pcVec(nIteration) = pc;
% %     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\% savedVariables\',dataSetName,'\',...
% %         folderName2,folderName3,'\Pd',num2str(nIteration)]);
% %     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\% savedVariables\',dataSetName,'\',...
% %         folderName2,folderName3,'\Pfa',num2str(nIteration)]);
% %     handle(5) = plotRocFun(Pfa,Pd,'LineWidth',2);
% %     hold all;
% end
% hold off;

% title('Musk1 ROC using same 10 Fold xVal Keys','FontSize',14);
% 
% legend('Default (K=20,20 \alpha= [0.5 0.5] \gamma_2 = 1e-3 gamInvScale = 5)',...
%         '100 Iterations Per Train Fold Max % Correct',...
%         '100 Iterations Per Train Fold Max NFE');
% 
% legend('Default (K=20,20 \alpha= [0.5 0.5] \gamma_2 = 1e-3 gamInvScale = 5)',...
%         'K=25,25',...
%         'K=30,30',...
%         'K=15,15',...
%         'K=10,10',...
%         'K=5,5');
%     
% legend('Default (K=20,20 \alpha= [0.5 0.5] \gamma_2 = 1e-3 gamInvScale = 5)',...
%         '\alpha= [47 424]',...
%         '\alpha= [1 9]\times1e+3');   
% 
% legend('Default (K=20,20 \alpha= [0.5 0.5] \gamma_2 = 1e-3 gamInvScale = 5)',...
%         '\gamma_2 = 1e-4',...
%         '\gamma_2 = 1e-6',...
%         '\gamma_2 = 1e-1');  
%     
% legend('Default (K=20,20 \alpha= [0.5 0.5] \gamma_2 = 1e-3 gamInvScale = 5)',...
%         'gamInvScale = 10',...
%         'gamInvScale = 1');    
        
% pc100 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-3 gamInvScale = 5
% pc101 10xValKeys K20/20 alpha [.5 .5] 100 Iterations Max Pc
% pc102 10xValKeys K20/20 alpha [.5 .5] 100 Iterations Max NFE
% pc103 10xValKeys K25/25 alpha [.5 .5]
% pc104 10xValKeys K30/30 alpha [.5 .5]
% pc105 10xValKeys K15/15 alpha [.5 .5]
% pc106 10xValKeys K10/10 alpha [.5 .5]
% pc107 10xValKeys K5/5 alpha [.5 .5]
% pc108 10xValKeys K20/20 alpha [47 424]
% pc109 10xValKeys K20/20 alpha [1 9]*1000
% pc110 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-4
% pc111 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-6
% pc112 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-1
% pc113 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-1 gamInvScale = 10
% pc114 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-1 gamInvScale = 100
% pc115 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-1 gamInvScale = 1

% musk2
% pc100 10xValKeys K20/20 alpha [.5 .5] gamma2 = 1e-3 gamInvScale = 5
% pc101 10xValKeys K20/20 alpha [.5 .5] 100 Iterations Max Pc
% pc102 10xValKeys K20/20 alpha [.5 .5] 100 Iterations Max NFE
% pc103 10xValKeys K25/25 alpha [.5 .5]
% pc104 10xValKeys K30/30 alpha [.5 .5]
% pc105 10xValKeys K15/15 alpha [.5 .5]
% pc106 10xValKeys K10/10 alpha [.5 .5]
% pc107 10xValKeys K5/5 alpha [.5 .5]
% pc108 10xValKeys K10/10 alpha [39 6533-39]
% pc109 10xValKeys K20/20 alpha [1 9]*1000
% pc110 10xValKeys K10/10 alpha [.5 .5] gamma2 = 1e-4
% pc111 10xValKeys K10/10 alpha [.5 .5] gamma2 = 1e-6
% pc112 10xValKeys K10/10 alpha [.5 .5] gamma2 = 1e-1
% pc113 10xValKeys K10/10 alpha [.5 .5] gamma2 = 1e-1 gamInvScale = 10
% pc114 10xValKeys K10/10 alpha [.5 .5] gamma2 = 1e-1 gamInvScale = 100
% pc115 10xValKeys K10/10 alpha [.5 .5] gamma2 = 1e-1 gamInvScale = 1
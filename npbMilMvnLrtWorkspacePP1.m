clear all;
close all;

pathToSave = 'C:\Users\manandhar\research\matlabStuffData\Research';

% ds = prtDataSetClass(bags.data,bags.label); explore(rt(prtPreProcPls('nComponents',5),ds));
% ds = prtDataSetClass(bags.data,bags.label); explore(run(train(prtPreProcPls('nComponents',5),ds),ds));

folderName1 = 'bnpMil';
dataSetName = 'ehd';
folderName2 = 'roc10xEhd6Pc';
folderName3 = 'NpbMilRndInit';

% folderName1 = 'npbMilOnline';
% dataSetName = 'ehd';
% folderName2 = 'roc10xEhd6Pc';
% folderName3 = 'NpbMilRndInitNby20';

% folderName1 = 'bnpMil';
% dataSetName = 'musk1';
% folderName2 = 'roc10xMusk1Pc20';
% folderName3 = 'VbMilFCRndK20';

% folderName1 = 'bnpMil';
% dataSetName = 'elephant';
% folderName2 = 'roc10xElephantNonSparsePc40';
% folderName3 = 'VbMilFCRndK20';

% folderName1 = 'bnpMil';
% dataSetName = 'msCorsid';
% folderName2 = 'roc10xMsCorsid';
% folderName3 = 'NpbMilFeatsUncorrRndInit';

% folderName1 = 'bnpMil';
% dataSetName = 'msrcorid';
% folderName2 = 'roc3xMsrcorid';
% folderName3 = 'NpbMilFeatsUncorrRndInit';

% folderName1 = 'bnpMil';
% dataSetName = 'msrcorid';
% % folderName2 = 'roc3xDsMilHog';
% % folderName2 = 'roc3xDsMilHogPc5';
% folderName2 = 'roc10xDsMilHogPc5';
% % folderName3 = 'NpbMilRndInit';
% folderName3 = 'NpbMilVbInit';
% % folderName3 = 'NpbMilFeatsUncorrRndInit';
% % folderName3 = 'NpbMilFeatsUncorrVbInit';

% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\msrcorid_Mil_Data');
% for nBag=1:length(dsMilHogFullStruct)
%     instThisBag = size(dsMilHogFullStruct(nBag).data,1);
%     dsMilHogFullStruct(nBag).label = bagLabels(nBag)*ones(instThisBag,1);
%     dsMilHogFullStruct(nBag).bagNum = nBag*ones(instThisBag,1);
% end
% bags = concatStructs(1,dsMilHogFullStruct);
%     
% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\msrcorid_Mil_Data');
% for nBag=1:length(dsMilHogPcaStruct)
%     instThisBag = size(dsMilHogPcaStruct(nBag).data,1);
%     dsMilHogPcaStruct(nBag).label = bagLabels(nBag)*ones(instThisBag,1);
%     dsMilHogPcaStruct(nBag).bagNum = nBag*ones(instThisBag,1);
% end
% bags = concatStructs(1,dsMilHogPcaStruct);

% load('C:\Users\manandhar\research\matlabStuff\Research\bnpMilSynDataExp\synDataBags200TwoH1TwoH0XorR1d25c0_5\datasetsTrain\bagsSet1.mat');bags=bagsTrain;
% load('bags200TwoH1TwoH0R2_5');
% load('C:\Users\manandhar\research\matlabStuffData\Research\milDatasets\musk1Norm');
% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\elephantNonSparse');

% load('hogMilBags');
load('C:\Users\manandhar\research\matlabStuffData\Research\milDatasets\EHD_HMDS');
% load('TFCM_HMDS');
% bags  = downSampleBags(bags,10); % make around 1000 H1 and 1000 H0
% bags = downSampleNegBags(bags,10);

% bags.data = zscore(bags.data);
% dataset = prtDataSetClass(zscore(bags.data),bags.label);
% pca = prtPreProcPca;
% pca.nComponents = 20;
% pca = pca.train(dataset); % default 3 PC
% datasetNew = pca.run(dataset);
% bags.data = datasetNew.data; % [totalTimeSamples * default 3 PC]
% nPc = 20;
% bags.data = myPrinCompFun(bags.data,nPc);
% save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%     folderName2,folderName3,'\pcaTrained'],'pca');
    
% lambda = factoran(bags.data,4);
% bags.data = bags.data*lambda;

% % Feature Selection
% dataSet = prtDataSetClass(bags.data,bags.label);
% featSel = prtFeatSelSfs;           % Create a feature selction object
% featSel.nFeatures = 10;            % Select only one feature of the data
% % featSel.showProgressBar = false;   % Progres bar
% featSel = featSel.train(dataSet);  % Train the feature selection object
% dataSetOut = featSel.run(dataSet); % Extract the data set with only the selected features
% bags.data = dataSetOut.getObservations;

% Bag labels, only one per bag
bagNums = unique(bags.bagNum);
totalBags = length(bagNums);
for nBag=1:totalBags
    myBagLabels(nBag,1) = unique(bags.label(bags.bagNum==bagNums(nBag),:));
end
    
for nIteration=1%:10
    
    myTrainFun          = @npbMilMvnTrain;
    myTestFun           = @npbMilMvnTest;
    myOptions.vbOnline          = false;
    myOptions.delay             = 1; % tau = 10 Don't down-weight early iterations quickly
    myOptions.forgettingRate    = .55; % kappa = {.5,.6,...,1.0} = .55 % Don't forget early iterations quickly
    myOptions.NBatches          = 20;
    myOptions.convergenceThreshold = 1e-6;
    myOptions.totalH1Clusters   = 10;
    myOptions.totalH0Clusters   = 10;
    myOptions.computeLogPred    = false;
    myOptions.plot              = 0;
    myOptions.featsUncorrelated = 0;
    myOptions.rndInit           = true;
    myOptions.maxIteration      = 1000;
    numOfFolds                  = 10;

    load(fullfile('C:\Users\manandhar\research\matlabStuffData\Research\milBenchmarkTime\nPC\ehdPc6',sprintf('10xValKeys%d.mat',nIteration)));
%     crossValKeys = myUtilRandEquallySubdivideData(1:length(unique(bags.bagNum)),10);
%     save(fullfile('C:\Users\manandhar\research\matlabStuffData\Research\milBenchmarkTime\nPC\ehdPc6',sprintf('10xValKeys%d.mat',nIteration)),'crossValKeys');
    myOptions.crossValKeys = crossValKeys;
%     profile on;
    [yOut, xValTrainedClassifier, crossValKeysByBag, timeTrain, timeTest] = nValMilTimePP1(bags,numOfFolds,myTrainFun,myTestFun,myOptions); % leave-one-bag-out xVal
%     profile report;
%     profile off; 

    % Percent Correct
    [pc, thresh]  = findPercentCorrect(myBagLabels,yOut);
    
    % ROC
    [Pfa,Pd,~,auc] = prtScoreRoc(yOut,myBagLabels);pcVec(nIteration)=pc;
    figure(4);
    handle1 = plotRocFun(Pfa,Pd,'color','b','LineWidth',2);
%     title([num2str(nPca),'-PCA Downsampled EHD, (',num2str(numOfFolds),'-Fold xVal, ',num2str(nIteration),' Iterations)'],'FontSize',10);  
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('timeTrain%d',nIteration)),'timeTrain');
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('timeTest%d',nIteration)),'timeTest');
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('Pd%d',nIteration)),'Pd');
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('Pfa%d',nIteration)),'Pfa');
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('auc%d',nIteration)),'auc');
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('pc%d',nIteration)),'pc');
    save(fullfile(pathToSave,folderName1,'savedVariables',dataSetName,strcat(folderName2,folderName3),sprintf('thresh%d',nIteration)),'thresh');   
    hold on;    
end
hold off;
beep;
pcVec

% folderName1 = 'bnpMil';
% dataSetName = 'musk2';
% folderName2 = 'roc10xMusk2';
% folderName3 = 'BnpMilFeatsUncorrRndInit';

% folderName1 = 'bnpMil';
% dataSetName = 'elephant';
% folderName2 = 'roc10xElephantNonSparseSumFeatsNot0';
% folderName3 = 'NpbMilVbInit';

% folderName1 = 'bnpMil';
% dataSetName = 'ehd';
% folderName2 = 'roc10xDs10Ehd6Pc';
% folderName3 = 'NpbMilRndInit';

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
%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\time',num2str(nIteration)]);
%     timeVec(nIteration) = timeForExecution;
% %     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
% %         folderName2,folderName3,'\auc',num2str(nIteration)]);
% %     aucVec(nIteration) = auc;
%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\pc',num2str(nIteration)]);
%     pcVec(nIteration) = pc;
%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\Pd',num2str(nIteration)]);
%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\Pfa',num2str(nIteration)]);
%     handle(5) = plotRocFun(Pfa,Pd,'LineWidth',2);
%     hold all;
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
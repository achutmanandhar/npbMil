clear all;
close all;

% ds = prtDataSetClass(bags.data,bags.label); explore(rt(prtPreProcPls('nComponents',5),ds));
% ds = prtDataSetClass(bags.data,bags.label); explore(run(train(prtPreProcPls('nComponents',5),ds),ds));

% folderName1 = 'bnpMil';
% dataSetName = 'trec1';
% folderName2 = 'roc10xTrec1SumFeatsGt0';
% folderName3 = 'NpbMilRndInit';

% folderName1 = 'bnpMil';
% dataSetName = 'trec2';
% folderName2 = 'roc10xTrec2SumFeatsGt0';
% folderName3 = 'NpbMilRndInit';

NWordsPerInstance = 50; % In fact, it should be different for every instance?
% parameters(1).P = [.4 .1 .4 .1]'; % H1
% parameters(2).P = [.1 .4 .1 .4]';
% parameters(3).P = [.1 .1 .4 .4]'; % H0
% parameters(4).P = [.4 .4 .1 .1]';
% d = 20;
% parameters(1).P = prtRvUtilDirichletDraw(ones(1,d),1); % H1
% parameters(2).P = prtRvUtilDirichletDraw(ones(1,d),1); 
% parameters(3).P = prtRvUtilDirichletDraw(ones(1,d),1); % H0
% parameters(4).P = prtRvUtilDirichletDraw(ones(1,d),1); 
% bags = MILgenerateMnSetTwoH1TwoH0(200,10,1,NWordsPerInstance,parameters);
% load('C:\Users\manandhar\research\matlabStuff\Research\vbDpMnMil\experimentSynDataTruthVsModel\bags400d1000TwoH1TwoH0\bagsFromTrueParam2.mat');
load('C:\Users\manandhar\research\matlabStuff\Research\bnpMilSynDataExp\synDataMnBags200TwoH1TwoH0d1000\datasetsTrain\bagsSet1');
bags = bagsTrain;

load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\trec2NonSparseSumFeatsGt0.mat');

for nIteration=1:10
    
    myTrainFun          = @npbMilMnTrain;
    myTestFun           = @npbMilMnTest;
    myOptions.totalH0Clusters = 10;
    myOptions.totalH1Clusters = 10;
    myOptions.plot = 0;
%     myOptions.featsUncorrelated = 0;
    myOptions.rndInit = 1;
%     myOptions.convergenceThreshold = 1e-5;
    numOfFolds          = 10;

%     load(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\10xValKeys1.mat']);
    crossValKeys = myUtilRandEquallySubdivideData(1:length(unique(bags.bagNum)),10);
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\10xValKeys',num2str(nIteration)],'crossValKeys');
    myOptions.crossValKeys = crossValKeys;
%     profile on;
tic;    
    yOut                = nValMil(bags,numOfFolds,myTrainFun,myTestFun,myOptions); % leave-one-bag-out xVal
timeForExecution = toc;    
%     profile report;
%     profile off; 

    % Bag labels, only one per bag
    bagNums = unique(bags.bagNum);
    totalBags = length(bagNums);
    for nBag=1:totalBags
        myBagLabels(nBag,1) = unique(bags.label(bags.bagNum==bagNums(nBag),:));
    end

    % Percent Correct
    [pc, thresh]  = findPercentCorrect(myBagLabels,yOut);
    
    % ROC
    [Pfa,Pd,~,auc] = prtScoreRoc(yOut,myBagLabels);pc
    figure(4);
    handle1 = plotRocFun(Pfa,Pd,'color','b','LineWidth',2);
%     title([num2str(nPca),'-PCA Downsampled EHD, (',num2str(numOfFolds),'-Fold xVal, ',num2str(nIteration),' Iterations)'],'FontSize',10);  
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\time',num2str(nIteration)],'timeForExecution');
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\Pd',num2str(nIteration)],'Pd');
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\Pfa',num2str(nIteration)],'Pfa');
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\auc',num2str(nIteration)],'auc');
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\pc',num2str(nIteration)],'pc');
%     save(['C:\Users\manandhar\research\matlabStuff\Research\',folderName1,'\savedVariables\',dataSetName,'\',...
%         folderName2,folderName3,'\thresh',num2str(nIteration)],'thresh');
    hold on;    
end
hold off;
beep;


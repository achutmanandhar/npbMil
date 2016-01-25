function outputsTrain = npbMilMvnTrain(Xtrain,myOptions)
% outputsTrain = npbMilTrain(Xtrain,myOptions)
%     myOptions.totalH0Clusters = 4;
%     myOptions.totalH1Clusters = 4;
%     myOptions.clusteringAlg = 'GMM';
%     myOptions.classificationAlg = 'SVM';

% Preproc Xtrain
ds = prtDataSetClass(Xtrain.data,Xtrain.label);
objZmuvPca = prtPreProcZmuv+prtPreProcPca('nComponents',6);
objZmuvPca = objZmuvPca.train(ds);
dsTrained = objZmuvPca.run(ds);
Xtrain.data = dsTrained.data;
outputsTrain.objZmuvPca = objZmuvPca;
% free memory
clear ds objZmuvPca dsTrained;
% If we are computing logPred density on test data, then will need preproc
if (myOptions.computeLogPred)
    myOptions.objZmuvPca = outputsTrain.objZmuvPca;
end

if (myOptions.vbOnline)
    outputsTrain.posteriors = npbMilMvnOnline(Xtrain,myOptions.totalH1Clusters,myOptions.totalH0Clusters,myOptions);
else
    outputsTrain.posteriors = npbMilMvn(Xtrain,myOptions.totalH1Clusters,myOptions.totalH0Clusters,myOptions);
end


% % Multiple Train
% for nIteration = 1:1
%     [post(nIteration).posteriors,post(nIteration).nfeTerms] = npbMilMvn(Xtrain,myOptions.totalH1Clusters,myOptions.totalH0Clusters,myOptions);   
%         post(nIteration).posteriors(1).pi
%         post(nIteration).posteriors(2).pi      
% %     nfeTermsCat = concatStructs(1,post(nIteration).nfeTerms);
% %     nfe(nIteration) = nfeTermsCat.nfe(end,1);
%     
%     yOut = npbMilMvnTest(Xtrain,[],post(nIteration),myOptions);
%     % Bag labels, only one per bag
%     bagNums = unique(Xtrain.bagNum);
%     totalBags = length(bagNums);
%     for nBag=1:totalBags
%         bagLabels(nBag,1) = unique(Xtrain.label(Xtrain.bagNum==bagNums(nBag),:));
%     end
%     % Percent Correct (how to determine
% %     [pc(nIteration),threshold(nIteration)]  = findPercentCorrect(bagLabels,yOut);
%     [Pfa,Pd,~,auc] = prtScoreRoc(yOut,bagLabels);
% end
% [~,index] = max(auc);
% % [~,index] = max(nfe);    
% outputsTrain = post(index);
% keyboard;

% posteriors(1).muN(:,1) = [2 2]';
% posteriors(1).muN(:,2) = [-2 -2]';
% posteriors(1).SigmaN(:,:,1) = eye(2);
% posteriors(1).SigmaN(:,:,2) = eye(2);
% posteriors(1).pi = [.5 .5];
% posteriors(1).lambda = [1 1];
% 
% posteriors(2).muN(:,1) = [-2 2]';
% posteriors(2).muN(:,2) = [2 -2]';
% posteriors(2).SigmaN(:,:,1) = eye(2);
% posteriors(2).SigmaN(:,:,2) = eye(2);
% posteriors(2).pi = [.5 .5];
% posteriors(2).lambda = [1 1];
% 
% posteriors(3).eta = [.1 .9];
% outputsTrain.posteriors = posteriors;
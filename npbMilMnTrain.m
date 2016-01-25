function outputsTrain = npbMilMnTrain(Xtrain,myOptions)
% outputsTrain = npbMilTrain(Xtrain,myOptions)
%     myOptions.totalH0Clusters = 4;
%     myOptions.totalH1Clusters = 4;
%     myOptions.clusteringAlg = 'GMM';
%     myOptions.classificationAlg = 'SVM';

outputsTrain.posteriors = npbMilMn(Xtrain,myOptions.totalH1Clusters,myOptions.totalH0Clusters,myOptions);

% % Multiple Train
% for nIteration = 1:1
%     [post(nIteration).posteriors,post(nIteration).nfeTerms] = npbMilMn(Xtrain,myOptions.totalH1Clusters,myOptions.totalH0Clusters,myOptions);   
%         post(nIteration).posteriors(1).pi
%         post(nIteration).posteriors(2).pi      
% %     nfeTermsCat = concatStructs(1,post(nIteration).nfeTerms);
% %     nfe(nIteration) = nfeTermsCat.nfe(end,1);
%     
%     yOut = npbMilMnTest(Xtrain,[],post(nIteration),myOptions);
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
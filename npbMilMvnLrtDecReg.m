clear all;
close all;

% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\bags200TwoGauss002_52_5I.mat');
load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\bags200TwoH1TwoH0R3.mat');

myOptions.totalH0Clusters = 20;
myOptions.totalH1Clusters = 20;
myOptions.plot = 1;
myOptions.featsUncorrelated = 0;
myOptions.rndInit = 0;         

nGrid = 200;
minXY = bsxfun(@minus,min(bags.data),1);
maxXY = bsxfun(@plus,max(bags.data),1);
xx = linspace(minXY(1),maxXY(1),nGrid);
yy = linspace(minXY(2),maxXY(2),nGrid);
[xGrid,yGrid] = meshgrid(xx,yy);

bags2DGrid.data         = cat(2,xGrid(:),yGrid(:));
bags2DGrid.bagNum(:,1)  = 1:nGrid*nGrid;

outputsTrain = npbMilMvnTrain(bags,myOptions);
outputsTest = npbMilMvnTest(bags2DGrid,[],outputsTrain,myOptions);
yOutTest    = reshape(outputsTest,size(xGrid));

handle=figure(52);
set(0,'Units','pixels') 
pos1 = [40 40 350 350];
set(handle,'OuterPosition',pos1)     
set(gcf, 'color', 'white');
imagesc(xx,yy,yOutTest);
set(gca,'Ydir','normal');
hold on;
plotTwoClusters2D(bags);
hold off;
title('Decision Region Using NPBMIL (K=20,20)','FontSize',12);
axis([minXY(1), maxXY(1), minXY(2), maxXY(2)]);

% s2('fig','C:\Users\manandhar\research\researchStuff\pptResults\2013\prelimStuffs\bags200TwoH1TwoH0R3DecRegPpmmK4');
% s2('png','C:\Users\manandhar\research\researchStuff\pptResults\2013\prelimStuffs\bags200TwoH1TwoH0R3DecRegPpmmK4');

% % after keyboard
% myOptions.pruneThreshold = .11;
% handle=figure(51);
% set(0,'Units','pixels') 
% pos1 = [40 40 700 350];
% set(handle,'OuterPosition',pos1)     
% set(gcf, 'color', 'white');
% 
% for mm=1:2
% predictives = zeros(size(gridSpace2D,1),1);
% K = length(post(mm).pi);
% for k=1:K
%     if (post(mm).pi(k)>myOptions.pruneThreshold)
%         if myOptions.featsUncorrelated              
%             predictives = predictives + post(mm).pi(k)*mvnpdf(gridSpace2D,post(mm).meanN(:,k)',post(mm).covN(:,k)'); % predictive densities (Gaussian)
%         else
%             predictives = predictives + post(mm).pi(k)*mvnpdf(gridSpace2D,post(mm).meanN(:,k)',post(mm).covN(:,:,k)); % predictive densities (Gaussian)
%         end
%     end
% %         predictives = predictives + (post(mm).lambda(k)/sumLambda)*pdfNonCentralT(gridSpace2D,post(mm).rho(:,k)',post(mm).Lambda(:,:,k),post(mm).degreesOfFreedom(k)); % predictive densities (t)
%     % or just use pi's instead of lambda's
% end
% gridSpace2DClusters = reshape(predictives,size(xGrid));
% 
% subplot(1,2,mm);
% imagesc(xx,yy,gridSpace2DClusters);
% % c = colormapRedblue;
% % colormap(c);
% % colorbar;
% % caxis([0 .1])
% set(gca,'Ydir','normal');
% hold on;
% 
% %     plot(bags.data(:,1),bags.data(:,2),'.r');
% plotTwoClusters2D(bags);
% hold on;
% for k=1:K
%     if (post(mm).pi(k)>myOptions.pruneThreshold)
%         if myOptions.featsUncorrelated
%             myPlotMvnEllipse(post(mm).meanN(:,k)',diag(post(mm).covN(:,k)),1,[],'b');
%         else
%             myPlotMvnEllipse(post(mm).meanN(:,k)',post(mm).covN(:,:,k),1,[],'b');
%         end
%     end
% %         plotMvnEllipse(post(mm).rho(:,k)',post(mm).Lambda(:,:,k)\eye(d),1);
%     % annotate contours with pi's
% end
% title(['H',num2str(MM(mm)),' Clusters (K=20)'],'FontSize',12);
% ylabel('Feature 2','FontSize',12);
% xlabel('Feature 1','FontSize',12);
% hold off;
% end


% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\EHD_HMDS');
% bags = downSampleBags(bags,10); % make around 1000 H1 and 1000 H0
% 
% bags.data = zscore(bags.data);
% dataset = prtDataSetClass(zscore(bags.data),bags.label);
% pca = prtPreProcPca;
% pca.nComponents = 2;
% pca = pca.train(dataset); % default 3 PC
% datasetNew = pca.run(dataset);
% bags.data = datasetNew.data; % [totalTimeSamples * default 3 PC]
%  handle=figure(20);
%     set(0,'Units','pixels') 
%     pos1 = [40 40 350 350];
%     set(handle,'OuterPosition',pos1)     
%     set(gcf, 'color', 'white'); 
%     plotTwoClusters2D(bags);
%     title('Landmine edge histogram descriptors (Frigui, 2009)');

% plot a neg bag and a pos bag synthetic data
% clear all;
% close all;
% load('C:\Users\manandhar\research\matlabStuff\Research\MILDataSets\bags200TwoGauss002_52_5I.mat');
% minXY = bsxfun(@minus,min(bags.data),1);
% maxXY = bsxfun(@plus,max(bags.data),1);
% 
% bagsNeg.data = bags.data(1:10,:);
% bagsNeg.label = bags.label(1:10,:);
% bagsPos.data = bags.data(501:510,:);
% bagsPos.label = bags.label(501:510,:);
% 
%     handle=figure(91);
%     set(0,'Units','pixels') 
%     pos1 = [40 40 250 250];
%     set(handle,'OuterPosition',pos1)     
%     set(gcf, 'color', 'white'); 
%     plotTwoClusters2D(bagsNeg);
%     axis([minXY(1), maxXY(1), minXY(2), maxXY(2)]);
%     hold on;
%     myPlotMvnEllipse([0 0],eye(2),1,[],'b');
%     myPlotMvnEllipse([3 3],eye(2),1,[],'b');
%     hold off;
%     title('A Y=0 Bag','FontSize',12); 
%     
%     handle=figure(92);
%     set(0,'Units','pixels') 
%     pos1 = [40 40 250 250];
%     set(handle,'OuterPosition',pos1)     
%     set(gcf, 'color', 'white'); 
%     plotTwoClusters2D(bagsPos);
%     axis([minXY(1), maxXY(1), minXY(2), maxXY(2)]);
%     hold on;
%     myPlotMvnEllipse([0 0],eye(2),1,[],'b');
%     myPlotMvnEllipse([3 3],eye(2),1,[],'b');
%     hold off;
%     title('A Y=1 Bag','FontSize',12); 

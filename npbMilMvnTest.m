function yOut = npbMilMvnTest(Xtest,Xtrain,outputsTrain,myOptions)
% outputsTest = classifyVbDpGmmMilLlrtTest(Xtest,Xtrain,outputsTrain,options)

if (isfield(outputsTrain,'objZmuvPca'))
    % Preproc Xtest
    ds = prtDataSetClass(Xtest.data);
    dsTest = outputsTrain.objZmuvPca.run(ds);
    Xtest.data = dsTest.data;
    % free memory
    clear ds dsTest;
end

post = outputsTrain.posteriors;

for mm=1:2
    logSumLikelihood(mm).values = npbMilLogSumLikelihood(Xtest,post(mm),myOptions);
end

% keyboard;
% handle = npbMilTestingCloserLook(Xtest,logSumLikelihood,post);

instancesBPosLogSumLikelihood = prtUtilSumExp(cat(2,log(post(1).eta(1))+logSumLikelihood(1).values,log(post(1).eta(2))+logSumLikelihood(2).values)')';
instancesClassLogSumLikelihood = cat(2,instancesBPosLogSumLikelihood,logSumLikelihood(2).values);

bagNums = unique(Xtest.bagNum);
totalBags = length(bagNums);
for nBag=1:totalBags
    instancesLoc = Xtest.bagNum==bagNums(nBag);
    bagLikelihoods(nBag,:) = sum(instancesClassLogSumLikelihood(instancesLoc,:),1);
end

% normalize
% bagLikelihoodsNorm = bagLikelihoods - repmat(prtUtilSumExp(bagLikelihoods,2),1,2); 
bagLikelihoodsNorm = bsxfun(@minus,bagLikelihoods,prtUtilSumExp(bagLikelihoods')');

yOut = bagLikelihoodsNorm(:,1)-bagLikelihoodsNorm(:,2);
end

function logSumlikelihoodHMM = npbMilLogSumLikelihood(bags,post,myOptions)
% Calculate likelihood given mth mixture model
    likelihoodHMM = zeros(size(bags.data,1),1);
    K = length(post.pi);
    for k=1:K
        if myOptions.featsUncorrelated
            likelihoodHMM = likelihoodHMM + post.pi(k)*mvnpdf(bags.data,post.meanN(:,k)',diag(post.covN(:,k))); % predictive densities (Gaussian)
        else
            likelihoodHMM = likelihoodHMM + post.pi(k)*mvnpdf(bags.data,post.meanN(:,k)',post.covN(:,:,k)); % predictive densities (Gaussian)
        end
%         likelihoodHMM = likelihoodHMM + post.pi(k)*pdfNonCentralT(bags.data,post.rho(:,k)',post.Lambda(:,:,k),post.degreesOfFreedom(k)); % predictive densities (t)
        if (any(isnan(likelihoodHMM)) || any(isinf(likelihoodHMM)))
            keyboard;
        end
    end
    likelihoodHMM(likelihoodHMM==0) = min(likelihoodHMM(likelihoodHMM~=0)); % POSSIBLE BUG!
    logSumlikelihoodHMM = log(likelihoodHMM);
end

function handle = npbMilTestingCloserLook(Xtest,logSumLikelihood,post)
    handle = figure(2);
    mmVector=[1 0];
    for mm=1:2
        subplot(2,1,mm);
        plot(logSumLikelihood(mm).values);grid on;grid minor;hold on;
        bagLabels = Xtest.label;
        bagLabels(Xtest.label==0|Xtest.label==-1,:)=min(logSumLikelihood(mm).values);
        bagLabels(Xtest.label==1,:)=max(logSumLikelihood(mm).values);
        plot(bagLabels,':k','LineWidth',2.5);
        xlabel('Instances in Test Bags','FontWeight','b');
        ylabel(['log sum \pi_{i} p(x_n|\Theta^{H_',num2str(mmVector(mm)),'})'],'FontWeight','b');
        hold off;
    end
%     figure(2);subplot(2,1,1);hist(logLikelihood(1).values);grid on;ylabel('p(x|H1)');subplot(2,1,2);hist(logLikelihood(2).values);grid on;ylabel('p(x|H0)');
end
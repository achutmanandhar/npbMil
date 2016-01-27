function [post, nfeTerms] = npbMilMvn(varargin)
% post = npbMil(varargin)
% 
% Based on W. D. Penny, "A Variational Bayesian Framework for d-dimensional Graphical
% Models"
%
% The notation follows that of Penny for the most part.
%
% Syntax:
%           posteriors = vbGmmCluster(bags.data)
%           posteriors = vbGmmCluster(bags.data,K)
%           posteriors = vbGmmCluster(bags.data,K, myOptions)
%
% Inputs:
%           bags.data = bags.data (N*d)
%           K = number of clusters = N (default)
%           myOptions.maxIteration = 5000 (default)
%           myOptions.convergenceThreshold = 1e-10 (default)
%           myOptions.pruneClusters = 1 (default)
%           myOptions.pruneThreshold = .02 (default)
%           myOptions.featsUncorrelated = 0 (default)

% keyboard;
[bags KH1 KH0 myOptions] = parseInputs(varargin);
% for bag labels consistency
bags.label(bags.label==-1,:) = 0;


    % grid space to plot decision region
    nGrid = 200;
    minXY = bsxfun(@minus,min(bags.data),1);
    maxXY = bsxfun(@plus,max(bags.data),1);
    xx = linspace(minXY(1),maxXY(1),nGrid);
    yy = linspace(minXY(2),maxXY(2),nGrid);
    [xGrid,yGrid] = meshgrid(xx,yy);
    bags2DGrid.data         = cat(2,xGrid(:),yGrid(:));
    bags2DGrid.bagNum(:,1)  = 1:nGrid*nGrid;


% initialize parameters
fprintf('Initializing parameters \n');
[prior,eStep]  = npbMilInitialization(bags,KH1,KH0,myOptions);

fprintf('Iterating VB Updates \n');
nIteration = 1; 
% frameCount = 1;
% pathToVideo = 'C:\Users\manandhar\research\matlabStuffData\Research\npbMil\videoSimDataDecReg';
converged  = 0;
while (~converged) 
    
    % M-step
    % Variational Posterior Approxiamtion of pi, mu, lamda (model parameters)
    post = npbMilMvnUpdate(bags,prior,eStep,myOptions);
    
%     % Plot clusters and decision region
%     handle = npbMilPlotLearnedClustersDecReg(bags,prior,post,nIteration,myOptions); 

    % Prune ineffective components
    if (myOptions.pruneClusters)
        [prior,post] = npbMilPrune(prior,post,myOptions);
    end

    % E-step
    eStep = npbMilMvnExpectation(bags,eStep,post,myOptions); % Need eStep for eStep??? 
    
    % Compute log-predicitive on test data
    if (myOptions.computeLogPred)
        logPredPerBag(nIteration,1) = toc;
        logPredPerBag(nIteration,2) = npbMilMvnLogPred(myOptions.Xtest,post,myOptions);
    end 
            
    % neg free energy
    nfeTerms(nIteration) = npbMilNegFreeEnergy(prior,post,eStep,myOptions);
    if (nIteration>1)
        nfeTerms(nIteration).nfe(:,2) = nfeTerms(nIteration).nfe(:,1)-nfeTerms(nIteration-1).nfe(:,1);
        if (nfeTerms(nIteration).nfe(:,1)<nfeTerms(nIteration-1).nfe(:,1))
            disp(['Warning - neg free energy decreased at nIteration=',num2str(nIteration)]);
        end
    end

    if (nIteration>2)
        nfeTermsCat = concatStructs(1,nfeTerms);
%         figure(1);plot(nfeTermsCat.nfe(:,1),'.');grid on;drawnow;
        [converged, differentialPercentage, signedPercentage] = isconverged(nfeTermsCat.nfe(nIteration),nfeTermsCat.nfe(nIteration-1),myOptions.convergenceThreshold);
        if (converged==1)
            fprintf('Convergence reached. Change in negative free energy below %.2e. \n',myOptions.convergenceThreshold);
        else
            fprintf('Negative Free Energy: %.2f, \t %.2e Change \n',nfeTermsCat.nfe(nIteration,1),nfeTermsCat.nfe(nIteration,2));
        end
    end

    % Plot learned clusters
    if (myOptions.plot)
        nfeTermsCat = concatStructs(1,nfeTerms);
        handle = npbMilPlotLearnedClusters(bags,prior,post,eStep,nfeTermsCat.nfe(:,1),nIteration,myOptions); % GMM model parameters (mu,Sigma) are Normal-Wishart distributed!
    end    
    
%     s2('jpg',fullfile(pathToVideo,sprintf('frame%.3d',frameCount)),'antiAlias',false);
  
%    frameCount=frameCount+1;     
    nIteration = nIteration+1;
    if (nIteration==myOptions.maxIteration+1)
        converged=1;
    end
end 
% keyboard;

% Prune ineffective components
if (myOptions.pruneClusters)
    [~,post] = npbMilPrune(prior,post,myOptions);
end

% Plot learned clusters
if (myOptions.plot)
    nfeTermsCat = concatStructs(1,nfeTerms);
    handle = npbMilPlotLearnedClusters(bags,prior,post,eStep,nfeTermsCat.nfe(:,1),nIteration,myOptions); % GMM model parameters (mu,Sigma) are Normal-Wishart distributed!
end

% keyboard;
% filePath='C:\Users\manandhar\research\matlabStuff\Research\npbMil\vbGmmMilSynDataBags200TwoH1TwoH0R3';
% movie2gif(frameMovie,filePath);

end

function [bags, KH1, KH0, myOptions] = parseInputs(inputCell)

if (isstruct(inputCell{1}))
    bags = inputCell{1};
    switch length(inputCell)
        case 1 % vbGmmCluster(bags)
            KH1 = 20;
            KH0 = 20;
            myOptions = struct;
        case 2 % vbGmmCluster(bags,KH1)
            KH1 = inputCell{2};
            KH0 = inputCell{2};
            myOptions = struct;
        case 3 % vbGmmCluster(bags,KH1,KH0)
            KH1 = inputCell{2};
            KH0 = inputCell{3};
            myOptions = struct;
        case 4 % vbGmmCluster(bags,KH1,KH0,myOptions)
            KH1 = inputCell{2};
            KH0 = inputCell{3};
            myOptions = inputCell{4};            
        otherwise
            error('Invalid number of input arguments');
    end
    % Make sure all fields are initialized
    if (~isfield(myOptions,'maxIteration'))
        myOptions.maxIteration = 5000;
    end
    if (~isfield(myOptions,'convergenceThreshold'))
        myOptions.convergenceThreshold = 1e-10;
    end
    if (~isfield(myOptions,'pruneClusters'))
        myOptions.pruneClusters = false;
    end
    if (~isfield(myOptions,'pruneThreshold'))
        myOptions.pruneThreshold = .02;
    end
    if (~isfield(myOptions,'featsUncorrelated'))
        myOptions.featsUncorrelated = false;
    end
    if (~isfield(myOptions,'rndInit'))
        myOptions.rndInit = true;
    end
    if (~isfield(myOptions,'plot'))
        myOptions.plot = false;
    end   
    if (~isfield(myOptions,'computeLogPred'))
        myOptions.computeLogPred = false;
    end      
    if (~isfield(myOptions,'forceOneH1InstancePerPosBag'))
        myOptions.forceOneH1InstancePerPosBag = false;
    end     
else
    error('Invalid input - bags should be a structure');
end
end

function [prior,eStep] = npbMilInitialization(bags,KH1,KH0,myOptions)
% [prior,eStep]  = npbMilInitialization(bags,KH1,KH1);
% prior(1) = H1 Mixture Model
% prior(2) = H0 Mixture Model
% prior(1) = H1H0 Mixture Model
%
% post(1) = H1 Mixture Model
% post(2) = H0 Mixture Model
% post(1) = H1H0 Mixture Model
% keyboard;
    [N,D] = size(bags.data);
    KH = [KH1 KH0];
    
    % Initialize Priors (god given)
    prior(1).alpha       = [.5 .5]; % eta~Dir(alpha) = mixing proportion of the H1/H0 mixture models
    for mm=1:2 % mm = mixture model
        K = KH(mm);
        prior(mm).pi = ones(1,K)/K;
        prior(mm).gamma1 = 1;          % v|pi~Beta(gamma1,gamma2)
        prior(mm).gamma2 = 1e-3;       % sparsity inversely prop to concentration parameter
        prior(mm).beta = D;    % mu|Gamma~N(meanMean,1/(beta*Gamma))
%         prior(mm).meanMean = zeros(D,K);
        prior(mm).meanMean = randn(D,K);
        if (myOptions.featsUncorrelated)
            prior(mm).gamShape = ones(D,1);   % tau~Gam(gamShape,gamInvScale)
            prior(mm).gamInvScale = 5*ones(D,1); % sparsity directly prop to inverse scale parameter
        else           
            prior(mm).dfCov = D;
            prior(mm).covCov = (D)*eye(D);             
        end        
    end         
    
    fuzzFactor = .1;
    % Initialize phi_l_m_M
    eStep(1).rH1H0 = eps*ones(N,2);
    eStep(1).rH1H0(bags.label==0,1) = fuzzFactor*rand(length(find(bags.label==0)),1); 
    eStep(1).rH1H0(bags.label==0,2) = 1-eps;
    eStep(1).rH1H0 = bsxfun(@rdivide,eStep(1).rH1H0,sum(eStep(1).rH1H0,2)); % Normalize
        
    % Initialize rH (small phi)
    % rnd init
    if (myOptions.rndInit) 
        for mm=1:2
            eStep(mm).rH = rand(N,KH(mm)); 
            % Normalize
            eStep(mm).rH = eStep(mm).rH ./ repmat(sum(eStep(mm).rH,2),1,KH(mm));
        end
    % vb init        
    else
        % Initialize H0 respobsibility matrix (small phi)
        % Cluster instances in B- bags
        [~,ctrsH0] = kmeans(bags.data(bags.label==0,:),KH0,'EmptyAction','singleton','MaxIter',100);
        distMatrixH0 = pdist2(bags.data,ctrsH0);
        [~,indexKH0] = min(distMatrixH0,[],2);
        eStep(2).rH = bsxfun(@eq,indexKH0,1:KH0);   % rH0 before VB

        % Weight H0 resp (small phi) by mix mod resp (big PHI)
        eStep(2).rHrH1H0 = bsxfun(@times,eStep(2).rH,eStep(1).rH1H0(:,2));
        
        % Update H0 Mix Model   
        post(2) = npbMilMvnUpdateThisMixMod(bags,prior(2),eStep(2),myOptions);         
        
        % Compute p(x|H0 MM) for all instances
        eStepThisMixMod = npbMilMvnExpectationThisMixMod(bags,post(2),eStep(2),eStep(1).rH1H0(:,2),myOptions);
        eStep(2).rH = eStepThisMixMod.rH;
        
%         instanceLogLikelihoodGivenHmodel = prtUtilSumExp(eStepThisMixMod.variationalAvgLL')';
        instanceLogLikelihoodGivenHmodel = npbMilInstanceLogLikelihoodGivenHmodel(bags,post(2),myOptions);
        
        % Find H1 instances
        bags.myLabelH = zeros(N,1);
        bags.myLabelH(bags.label==1 & instanceLogLikelihoodGivenHmodel<min(instanceLogLikelihoodGivenHmodel(bags.label==0)),1) = 1;

        bagNums = unique(bags.bagNum);
        totalBags = length(bagNums);
        for nBag=1:totalBags
            if (unique(bags.label(bags.bagNum==bagNums(nBag))) == 1)
                % enforce at least one H1 instance per positive bag
                if ~any(bags.myLabelH(bags.bagNum==bagNums(nBag)))
                    [minVal,idx] = min(instanceLogLikelihoodGivenHmodel(bags.bagNum==bagNums(nBag)));
                    bags.myLabelH(bags.bagNum==bagNums(nBag) & instanceLogLikelihoodGivenHmodel==minVal,1) = 1;
                end
            end
        end
        
        % Initialize H1 respobsibility matrix
        % Cluster H1 instances
        [~,ctrsH1] = kmeans(bags.data(bags.myLabelH==1,:),KH(mm),'EmptyAction','singleton');

        % Initialize rH1
        distMatrixH1 = pdist2(bags.data,ctrsH1);
        [~,indexKH1] = min(distMatrixH1,[],2);
        eStep(1).rH = bsxfun(@eq,indexKH1,1:KH(mm));   % rH1
        
        % Initialize rH1H0 (big PHI)
        eStep(1).rH1H0 = zeros(N,2);
        eStep(1).rH1H0(bags.myLabelH==1,1) = 1;
        eStep(1).rH1H0(bags.myLabelH==0,2) = 1;

        % Weight small phi by big PHI
        eStep(1).rHrH1H0 = bsxfun(@times,eStep(1).rH,eStep(1).rH1H0(:,1));
        
        % Update H1 Mix Model
        post(1) = npbMilMvnUpdateThisMixMod(bags,prior(1),eStep(1),myOptions);
        
        % Compute p(x|H0 MM) for all instances
        eStepThisMixMod = npbMilMvnExpectationThisMixMod(bags,post(1),eStep(1),eStep(1).rH1H0(:,1),myOptions);
        eStep(1).rH = eStepThisMixMod.rH;
    end
% figure(1);imagesc(eStep(1).rH1H0);
% title('Big Responsibilty Matrix','FontSize',14)
  
    % weight small phi's by big PHI
    for mm=1:2
        eStep(mm).rHrH1H0 = bsxfun(@times,eStep(mm).rH,eStep(1).rH1H0(:,mm));
    end 
   
end

function instanceLogLikelihoodGivenHmodel = npbMilInstanceLogLikelihoodGivenHmodel(bags,post,myOptions)
%keyboard;
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
    instanceLogLikelihoodGivenHmodel = log(likelihoodHMM);
end

function post = npbMilMvnUpdate(bags,prior,eStep,myOptions)
% keyboard;
% 25/01/2016 AM
% In prtMilBrvMixture, need to add the following steps
    for mm=1:2
        post(mm) = npbMilMvnUpdateThisMixMod(bags,prior(mm),eStep(mm),myOptions);
    end    
% 25/01/2016 AM
% In prtMilBrvMixture, mixing proportion should be taken care by an object in prtBrvDiscrete
    post(1).countsH   = sum(eStep(1).rH1H0(bags.label==1,:),1);   
    post(1).countsH(post(1).countsH==0) = eps;
    post(1).alpha  = post(1).countsH + prior(1).alpha;
    post(1).eta = post(1).alpha/sum(post(1).alpha,2); 
end

function post = npbMilMvnUpdateThisMixMod(bags,prior,eStep,myOptions)
% keyboard;
    [~,D] = size(bags.data);
    K = size(prior.meanMean,2);
    
    % 25/01/2016 AM
    % In prtBrvMixture, resp or weights is known as training.componentMemberships
    % i.e. in prtMilBrvMixture, training.componentMemberships should be
    % weighted by rHr
    resp = eStep.rHrH1H0;   
    
    % 25/01/2016 AM
    % In prtBrvMixture, mixing proportions are defined by class prtBrvDiscrete
    % How does training.componentMemberships affect mixing propertions'
    % updates?
    post.counts = sum(resp,1);
    post.counts(post.counts==0) = eps;
    counts = post.counts;

    % mixing
    % sort counts
    sortObj = vbMilSortCount(counts); % If exchangeable, why sort? Why does sorting help?
    
    post.gamma1 = prior.gamma1 + sortObj.sortedCounts;
    post.gamma2 = prior.gamma2 + fliplr(cumsum(fliplr(sortObj.sortedCounts),2)) - sortObj.sortedCounts;
    post.pi = counts/sum(counts);
        
	% obs model
    post.beta = counts + prior.beta;
    if (myOptions.featsUncorrelated)      
        post.gamShape = bsxfun(@plus,repmat(counts/2,D,1),prior.gamShape);
    else
        post.dfCov = counts + prior.dfCov;
    end    

    for k=1:K                     
        muBar(:,k) = (sum(bsxfun(@times,resp(:,k),bags.data),1)/counts(k))';
        YMinusMu   = bsxfun(@minus,bags.data,muBar(:,k)');
        if (myOptions.featsUncorrelated==1)
            sigmaBar(:,k) = sum(bsxfun(@times,resp(:,k),YMinusMu.^2),1) / counts(k);
        else
            sigmaBar(:,:,k) = YMinusMu' * bsxfun(@times,resp(:,k),YMinusMu) / counts(k);
        end

        post.meanMean(:,k) = (prior.beta * prior.meanMean(:,k) + counts(k)*muBar(:,k)) / post.beta(k); % eq (21)                            
        sampleMeanMinusMeanMean = muBar(:,k) - prior.meanMean(:,k);        
        if (myOptions.featsUncorrelated==1)          
            post.gamInvScale(:,k) = prior.gamInvScale + .5*counts(k)*sigmaBar(:,k) + ...
                .5*(prior.beta*counts(k)/(post.beta(k))) * sum(sampleMeanMinusMeanMean.^2,2); % eq (23)
        else            
            post.covCov(:,:,k) = prior.covCov + counts(k)*sigmaBar(:,:,k) + ...
                (prior.beta*counts(k)/(prior.beta+counts(k))) * sampleMeanMinusMeanMean * (sampleMeanMinusMeanMean'); % eq (23)
        end
               
        % Mean, covariance
        post.meanN(:,k) = post.meanMean(:,k)'; 
        if (myOptions.featsUncorrelated==1)
            post.covN(:,k) = post.gamInvScale(:,k)./post.gamShape(:,k); 
        else
            post.covN(:,:,k) = post.covCov(:,:,k)/post.dfCov(k);
            if ~(isCovPosSemiDef(post.covN(:,:,k)))
                post.covN(:,:,k) = roundn(post.covN(:,:,k),-4);
            end
        end       
        
        % Find parameters of the Student t-distribution
        if (myOptions.featsUncorrelated==1)
            post.degreesOfFreedom(k) = post.gamShape(k) + 1 - D;
            post.Lambda(:,:,k) = ( (post.beta(k) + 1)/(post.beta(k)*post.degreesOfFreedom(k)) ) * post.gamInvScale(:,k); 
        else
            post.degreesOfFreedom(k) = post.dfCov(k) + 1 - D;
            post.Lambda(:,:,k) = ( (post.beta(k)*post.degreesOfFreedom(k))/(post.beta(k) + 1) ) * (post.covCov(:,:,k)\eye(D));
        end
    end
  
end

function eStep = npbMilMvnExpectation(bags,eStep,post,myOptions)
% keyboard;
    eStep(1).logEtaTilda    = psi(0,post(1).alpha) - psi(0,sum(post(1).alpha));
%     for nIter=1:5
        for mm=1:2
            eStepThisMixMod = npbMilMvnExpectationThisMixMod(bags,post(mm),eStep(mm),eStep(1).rH1H0(:,mm),myOptions);
            eStep(mm).rH = eStepThisMixMod.rH;
            eStep(mm).rHrH1H0 = eStepThisMixMod.rHrH1H0;
            eStep(mm).variationalAvgLL = eStepThisMixMod.variationalAvgLL;
            eStep(mm).logPiTilda = eStepThisMixMod.logPiTilda;
            eStep(1).logrH1H0Tilda(:,mm) = sum(eStep(mm).rH .* eStep(mm).variationalAvgLL,2) + eStep(1).logEtaTilda(mm); % [N*1]
        end    
        
        % Normalize
        eStep(1).rH1H0 = exp(bsxfun(@minus,eStep(1).logrH1H0Tilda,prtUtilSumExp(eStep(1).logrH1H0Tilda')')); % eq (15)

            % If a sample is not resposible to anyone randomly assign him
            % Ideally if initializations were good this would never happen. 
            noResponsibilityPoints = find(sum(eStep(1).rH1H0,2)==0);
            if ~isempty(noResponsibilityPoints)
                for iNrp = 1:length(noResponsibilityPoints)
                    eStep(1).rH1H0(noResponsibilityPoints(iNrp),:) = .5;
                end
            end

        if any(isnan(eStep(mm).rH1H0))
            keyboard;
        end    
        
        % enforce correct responsibilities for instances in B- bags     
        eStep(1).rH1H0(bags.label==0,1) = eps;    
        eStep(1).rH1H0(bags.label==0,2) = 1-eps; 
        % Make sure log(rH1H0) wont be infs
        eStep(1).rH1H0(eStep(1).rH1H0==0) = eps;
        eStep(1).rH1H0(eStep(1).rH1H0==1) = 1-eps;

%         % Hack!
%         % force at least one H1 instance per positive bag
%         if myOptions.forceOneH1InstancePerPosBag
%             bagNums = unique(bags.bagNum);
%             totalBags = length(bagNums);
%             for nBag=1:totalBags
%                 isThisBag = bags.bagNum==bagNums(nBag);
%                 thisBagIndices = find(isThisBag);
%                 if unique(bags.label(thisBagIndices))==1
%                     if ~any(eStep(1).rH1H0(thisBagIndices,1))
%                         % enforce
%                         [~,idx] = max(mixModLogLikelihoods(thisBagIndices,1));
%                         eStep(1).rH1H0(thisBagIndices(idx),1) = 1;
%                         eStep(1).rH1H0(thisBagIndices(idx),2) = 0;
%                     end
%                 end
%             end
%         end
%     end

    % weight small phi's by big PHI
    for mm=1:2
        eStep(mm).rHrH1H0 = bsxfun(@times,eStep(mm).rH,eStep(1).rH1H0(:,mm));
    end  
end

function eStep = npbMilMvnExpectationThisMixMod(bags,post,eStep,rH1H0ThisMixMod,myOptions)
% keyboard;
    [N,D] = size(bags.data);
    dims = 1:D;
    K = length(post.pi);
    logGammaTilda     = nan(1,K);
    variationalAvgLL  = nan(N,K);

    % E-Step    
    % mix
    eStep.logVTildaSorted            = psi(0,post.gamma1) - psi(0,post.gamma1+post.gamma2);
    eStep.log1MinusVTildaSorted      = psi(0,post.gamma2) - psi(0,post.gamma1+post.gamma2);
    for k=1:K
        if (k==1)
            logPiTildaSorted(k)        = eStep.logVTildaSorted(k);
        else
            logPiTildaSorted(k)        = eStep.logVTildaSorted(k) + sum(eStep.log1MinusVTildaSorted(1:k-1),2);
        end
    end    
    % Unsort
    sortObj = vbMilSortCount(post.counts);
    logPiTilda = logPiTildaSorted(sortObj.unsortIndices);
        
    % model
    for k=1:K        
        YMinusmeanMean           = bsxfun(@minus,bags.data,post.meanMean(:,k)');      
        if (myOptions.featsUncorrelated)
            eStep.logTauTilda(:,k)    = psi(0,post.gamShape(:,k)/2) - log(post.gamInvScale(:,k)); 
            expectMahalDist     = YMinusmeanMean.^2 * (post.gamShape(:,k)./post.gamInvScale(:,k)) + 1/post.beta(k);
%             expectMahalDist     = sum(bsxfun(@times,(post.gamShape(:,k)./post.gamInvScale(:,k))',YMinusmeanMean.^2) + 1/post.beta(k),2);
            variationalAvgLL(:,k)     = -0.5*D*log(2*pi) + 0.5*sum(eStep.logTauTilda(:,k),1) - expectMahalDist;
        else
            logGammaTilda(:,k)      = sum( psi(0,(post.dfCov(:,k)+1-dims)/2) ) - prtUtilLogDet(post.covCov(:,:,k)) + D*log(2);
            GammaBarInv    = post.covCov(:,:,k)/post.dfCov(k) + 1e-4; % 1e-4 for regularization
                if ~(isCovPosSemiDef(GammaBarInv))
                    GammaBarInv = roundn(GammaBarInv,-4);
                end
            expectMahalDist  = prtUtilCalcDiagXcInvXT(YMinusmeanMean,GammaBarInv) + D/post.beta(k);
                if (any(expectMahalDist<0)) % distance can't be negative
                    disp('Error - distance cannot be negative (set to eps)');
                    keyboard;
        %                 expectMahalDist(expectMahalDist(:,k)<0,k)=eps;
                end
            variationalAvgLL(:,k)     = 0.5*(-D*log(2*pi) + logGammaTilda(k) - expectMahalDist);
        end
%         if any((variationalAvgLL(:,k)==-inf|variationalAvgLL(:,k)==0))
%             keyboard;
% %             variationalAvgLL(variationalAvgLL==-inf|variationalAvgLL==0)=min(variationalAvgLL(variationalAvgLL~=0));
%         end
%         logrHTilda(:,k)   = rH1H0ThisMixMod.*variationalAvgLL(:,k) + logPiTilda(k);
    end
  
    if ( any(isinf(variationalAvgLL(:))) || any(variationalAvgLL(:)==0) )
        keyboard;
        variationalAvgLL(variationalAvgLL==-inf|variationalAvgLL==0)=min(variationalAvgLL(variationalAvgLL~=0));
    end
    
    % 25/01/2016 AM
    % In contrast to the eStep in standard prtBrvMvn, rHTilda is weighted by rH1H0ThisMixMod
    logrHTilda = bsxfun( @plus,logPiTilda,bsxfun(@times,rH1H0ThisMixMod,variationalAvgLL) );  % sim to eq (13) % [N*1]

    if ( any(isinf(logrHTilda(:,k))) || any(isnan(logrHTilda(:,k))) ) % check for inf/nan
        disp('Error - inf/nan in logResp values');
        keyboard;
    end
        
    % Normalize
    eStep.rH = exp(bsxfun(@minus,logrHTilda,prtUtilSumExp(logrHTilda')'));
            
    % If a sample is not resposible to anyone randomly assign him
    % Ideally if initializations were good this would never happen. 
    noResponsibilityPoints = find(sum(eStep.rH,2)==0);
    if ~isempty(noResponsibilityPoints)
        for iNrp = 1:length(noResponsibilityPoints)
            eStep.rH(noResponsibilityPoints(iNrp),:) = 1/K;
        end
    end
    if any(isnan(eStep.rH))
        keyboard;
    end    
    eStep.logPiTilda = logPiTilda;
    eStep.variationalAvgLL = variationalAvgLL;
    eStep.logrHTilda = logrHTilda;
end

function nfeTerms = npbMilNegFreeEnergy(prior,post,eStep,myOptions)
% keyboard;
% Based on
% W. D. Penny, "A Variational Bayesian Framework for d-dimensional Graphical
% Models"
% W. D. Penny, "KL-Divergences of Normal, Gamma, Dirichlet and Wishart
% Densities"
%
    % Calculation of Lav term in the negative free energy equation eq(25)
    KLz = zeros(1,2);
    variationalAvgLL = zeros(1,2);
    for mm=1:2
        variationalAvgLL(mm) = sum(sum(eStep(mm).rHrH1H0 .* eStep(mm).variationalAvgLL,1),2);

        respSmallzInfoTerm = eStep(mm).rHrH1H0.*log(eStep(mm).rH);
        respSmallzInfoTerm(eStep(mm).rHrH1H0==0) = 0;

        expectLogPiTerm = post(mm).counts.*eStep(mm).logPiTilda;
        KLz(mm) = sum( (sum(respSmallzInfoTerm,1) - expectLogPiTerm), 2);
%         KLz(mm) = sum( sum(respSmallzInfoTerm,1) - suffStat(mm).counts.*eStep(mm).logPiTilda, 2);
    end

    respOfMixModInfoTerm = eStep(1).rH1H0.*log(eStep(1).rH1H0);
    respOfMixModInfoTerm(eStep(1).rH1H0==0)=0;
    KLZeta = sum(respOfMixModInfoTerm) - post(1).countsH.*eStep(1).logEtaTilda;
    Lav = sum(variationalAvgLL - KLZeta - KLz, 2);

    sumKLZetaTerm = -sum(KLZeta);
    sumKLzTerm = -sum(KLz);
    sumExpectLLTerm = sum(variationalAvgLL);    
    sumKLeta = KLDirichlet(post(1).alpha,prior(1).alpha);

    mthKLv = zeros(1,2);
    mthKLMuAndGamma = zeros(1,2);
    for mm=1:2 % mm = nth mixture model
        K = length(post(mm).pi);
        KLv       = zeros(1,K);
        KLMuAndGamma = zeros(1,K);
        for k=1:K
            KLv(k) = KLBeta(post(mm).gamma1(k),post(mm).gamma2(k),prior(mm).gamma1,prior(mm).gamma2);
            if (myOptions.featsUncorrelated)
                KLMuAndGamma(k) = KLNormalGamma(post(mm).meanMean(:,k),prior(mm).meanMean(:,k),post(mm).beta(k),prior(mm).beta,post(mm).gamShape(:,k),prior(mm).gamShape,post(mm).gamInvScale(:,k),prior(mm).gamInvScale);
            else               
                KLMuAndGamma(k) = KLNormalWishart(post(mm).meanMean(:,k),prior(mm).meanMean(:,k),post(mm).beta(k),prior(mm).beta,post(mm).dfCov(k),prior(mm).dfCov,post(mm).covCov(:,:,k),prior(mm).covCov);
            end            
        end
        mthKLv(mm) = sum(KLv);
        mthKLMuAndGamma(mm) = sum(KLMuAndGamma);
    end

    sumKLv = sum(mthKLv);
    sumKLMuAndGamma = sum(mthKLMuAndGamma);
    F = Lav - sumKLeta - sumKLv - sumKLMuAndGamma;

    if (isinf(F))
        disp('Error! Neg Free Nergy is INF!');
    %     keyboard;
    end

    nfeTerms.nfe        = zeros(1,2);
    nfeTerms.nfe(:,1)   = F;
    nfeTerms.Lav        = sumExpectLLTerm;
    nfeTerms.KLZeta     = sumKLZetaTerm;
    nfeTerms.KLz        = sumKLzTerm;
    nfeTerms.KLeta      = sumKLeta;
    nfeTerms.KLv       = sumKLv;
    nfeTerms.KLMuAndGamma = sumKLMuAndGamma;
end

function logPredPerBag = npbMilMvnLogPred(bags,post,myOptions)    
keyboard;
    % Preproc Xtest
    ds = prtDataSetClass(bags.data,bags.label);
    dsTest = myOptions.objZmuvPca.run(ds);
    bags.data = dsTest.data;
    % free memory
    clear ds dsTest;

    % instance LL
    instanceLogLikelihoodGivenHmodel = nan(size(bags.data,1),2);
    for mm=1:2
        instanceLogLikelihoodGivenHmodel(:,mm) = npbMilInstanceLogLikelihoodGivenHmodel(bags,post(mm),myOptions);
    end
    instancesLogSumLikelihood = prtUtilSumExp(bsxfun(@plus,log(post(1).eta),instanceLogLikelihoodGivenHmodel)')';
%     logPred = sum(instancesLogSumLikelihood);
    instancesClassLogSumLikelihood = cat(2,instancesLogSumLikelihood,instanceLogLikelihoodGivenHmodel(:,mm));
    bagNumsReordered = grp2idx(bags.bagNum);
    logPredPerBag = mean(accumarray(bagNumsReordered,sum(instancesClassLogSumLikelihood,2)));  
end

function [prior post] = npbMilPrune(prior,post,myOptions)
% keyboard;
    for mm=1:2
        while any(post(mm).pi<myOptions.pruneThreshold)
            for k=1:length(post(mm).pi)
                if (post(mm).pi(k)<myOptions.pruneThreshold)   
                    prior(mm).meanMean(:,k)  = [];

                    post(mm).counts(k) = [];
                    post(mm).gamma1(k)  = [];
                    post(mm).gamma2(k)  = [];
                    post(mm).pi(k)      = [];
                    post(mm).beta(k)    = [];                    
                    post(mm).meanMean(:,k)   = [];
                    if myOptions.featsUncorrelated
                        post(mm).gamShape(:,k)     = [];
                        post(mm).gamInvScale(:,k)= [];
                    else
                        post(mm).dfCov(k)     = [];
                        post(mm).covCov(:,:,k)= [];
                    end

                    post(mm).degreesOfFreedom(k) = [];
                    post(mm).Lambda(:,:,k) = [];
                    post(mm).meanN(:,k)      = [];
                    if myOptions.featsUncorrelated
                        post(mm).covN(:,k)= [];
                    else
                        post(mm).covN(:,:,k)= [];
                    end                    
                    break;
                end
            end
        end
    end
end

function fig1 = npbMilPlotLearnedClusters(bags,prior,post,eStep,negFreeEnergy,nIteration,myOptions)
% keyboard;
[~,post] = npbMilPrune(prior,post,myOptions);
    fig1 = figure(1);
    d = size(bags.data,2);
    
    if (d==2)
        nGrid   = 200;
        tempMin = min(bags.data);
        xMin    = tempMin(1)-1;
        yMin    = tempMin(2)-1;
        tempMax = max(bags.data);
        xMax    = tempMax(1)+1;
        yMax    = tempMax(2)+1;

        xx = linspace(xMin,xMax,nGrid);
        yy = linspace(yMin,yMax,nGrid);
        [xGrid yGrid] = meshgrid(xx,yy);

        gridSpace2D = cat(2,xGrid(:),yGrid(:));

        MM = [1 0];
        for mm=1:2
            predictives = zeros(size(gridSpace2D,1),1);
            K = length(post(mm).pi);
            for k=1:K
                if (post(mm).pi(k)>myOptions.pruneThreshold)
                    if myOptions.featsUncorrelated              
                        predictives = predictives + post(mm).pi(k)*mvnpdf(gridSpace2D,post(mm).meanN(:,k)',post(mm).covN(:,k)'); % predictive densities (Gaussian)
                    else
                        predictives = predictives + post(mm).pi(k)*mvnpdf(gridSpace2D,post(mm).meanN(:,k)',post(mm).covN(:,:,k)); % predictive densities (Gaussian)
                    end
                end
        %         predictives = predictives + (post(mm).lambda(k)/sumLambda)*pdfNonCentralT(gridSpace2D,post(mm).rho(:,k)',post(mm).Lambda(:,:,k),post(mm).degreesOfFreedom(k)); % predictive densities (t)
                % or just use pi's instead of lambda's
            end
            gridSpace2DClusters = reshape(predictives,size(xGrid));

            %%%%%%%%%%%%%%%%%%%%%%
%             set(0,'Units','pixels') 
%             scnsize = get(0,'ScreenSize');
% 
%             position = get(fig1,'Position');
%             outerpos = get(fig1,'OuterPosition');
%             borders = outerpos - position;
% 
%             edge = -borders(1)/2;
%             pos1 = [edge,...
%                     scnsize(4) * (1/5),...
%                     scnsize(3)*(5/10) - edge,...
%                     scnsize(4)*(6/10)];
%             set(fig1,'OuterPosition',pos1)     
%             set(gcf, 'color', 'white');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%

            subplot(2,3,mm);
            imagesc(xx,yy,gridSpace2DClusters);
            % c = colormapRedblue;
            % colormap(c);
            % colorbar;
            % caxis([0 .1])
            set(gca,'Ydir','normal');
            hold on;

        %     plot(bags.data(:,1),bags.data(:,2),'.r');
            plotTwoClusters2D(bags);
            hold on;
            for k=1:K
                if (post(mm).pi(k)>myOptions.pruneThreshold)
                    if myOptions.featsUncorrelated
                        myPlotMvnEllipse(post(mm).meanN(:,k)',diag(post(mm).covN(:,k)),1,[],'b',2);
                    else
                        myPlotMvnEllipse(post(mm).meanN(:,k)',post(mm).covN(:,:,k),1,[],'b',2);
                    end
                end
        %         plotMvnEllipse(post(mm).rho(:,k)',post(mm).Lambda(:,:,k)\eye(d),1);
                % annotate contours with pi's
            end
            title(['H',num2str(MM(mm)),' Clusters (nIteration=',num2str(nIteration),')'],'FontWeight','b','FontSize',12);
            ylabel('Feature 2','FontWeight','b','FontSize',12);
            xlabel('Feature 1','FontWeight','b','FontSize',12);
            hold off;
        end
    end

    colorVector = ['r','b'];
    subplot(2,3,4);
    Kmax = max([length(post(1).pi) length(post(2).pi)]);
    for mm=1:2
        stem(post(mm).pi,'fill','MarkerFaceColor',colorVector(mm),'Color',colorVector(mm),'LineWidth',2); grid on; grid minor;
        axis([1 Kmax+1 0 1]);
        hold on;
    end
    ylabel('\pi_i','FontWeight','b','FontSize',14);
    xlabel('components in a mixture model','FontWeight','b','FontSize',12);
    legend('{\pi_i}^{1}','{\pi_i}^{0}','location','best');
    hold off;

    subplot(2,3,5);
    stem(post(1).eta,'fill','MarkerFaceColor',colorVector(mm),'Color','k','LineWidth',2); grid on;
    axis([1 2 0 1]);
    hold on;
    ylabel('\eta_{m}','FontWeight','b','FontSize',14);
    xlabel('mixture models','FontWeight','b','FontSize',12);
    hold off;

    % subplot(2,2,5); 
    % imagesc(eStep(1).rH); 
    % title(['\phi^{1}_{li}'],'FontWeight','b');%colorbar;
    % ylabel('instances','FontWeight','b');
    % xlabel('K^{1}','FontWeight','b');
    % 
    % subplot(2,2,6); 
    % imagesc(eStep(2).rH); 
    % title(['\phi^{0}_{li}'],'FontWeight','b');%colorbar;
    % ylabel('instances','FontWeight','b');
    % xlabel('K^{0}','FontWeight','b');
    % 
    % subplot(2,2,7); 
    % imagesc(eStep(1).rH1H0); 
    % title(['\phi^{M}_{lm}'],'FontWeight','b');%colorbar;
    % ylabel('instances','FontWeight','b');
    % xlabel('m=\{1,0\}','FontWeight','b');
    % 
    subplot(2,3,6);
    plot(negFreeEnergy,'.'); grid on;
    ylabel('Neg Free Energy','FontWeight','b');
    xlabel('nIteration','FontWeight','b');
    
    drawnow;
end

function fig1 = npbMilPlotLearnedClustersDecReg(bags,prior,post,nIteration,myOptions)
    if (nIteration>1)
        [~,post] = npbMilPrune(prior,post,myOptions);    
    end
    d = size(bags.data,2);
    
    bagsDS = downSampleBags(bags,5);
    
    fig1 = figure(801);
    set(0,'Units','pixels') 
    pos1 = [680 400 700 600];
    set(fig1,'OuterPosition',pos1);
    set(gcf, 'color', 'white'); 
    
    if (d==2)
        nGrid   = 200;
        tempMin = min(bags.data);
        xMin    = tempMin(1)-1;
        yMin    = tempMin(2)-1;
        tempMax = max(bags.data);
        xMax    = tempMax(1)+1;
        yMax    = tempMax(2)+1;

        xx = linspace(xMin,xMax,nGrid);
        yy = linspace(yMin,yMax,nGrid);
        [xGrid yGrid] = meshgrid(xx,yy);

        gridSpace2D = cat(2,xGrid(:),yGrid(:));

        MM = [1 0];
        for mm=1:2
            predictives = zeros(size(gridSpace2D,1),1);
            K = length(post(mm).pi);
            for k=1:K
                if (post(mm).pi(k)>myOptions.pruneThreshold)
                    if myOptions.featsUncorrelated              
                        predictives = predictives + post(mm).pi(k)*mvnpdf(gridSpace2D,post(mm).meanN(:,k)',post(mm).covN(:,k)'); % predictive densities (Gaussian)
                    else
                        predictives = predictives + post(mm).pi(k)*mvnpdf(gridSpace2D,post(mm).meanN(:,k)',post(mm).covN(:,:,k)); % predictive densities (Gaussian)
                    end
                end
        %         predictives = predictives + (post(mm).lambda(k)/sumLambda)*pdfNonCentralT(gridSpace2D,post(mm).rho(:,k)',post(mm).Lambda(:,:,k),post(mm).degreesOfFreedom(k)); % predictive densities (t)
                % or just use pi's instead of lambda's
            end
            gridSpace2DClusters = reshape(predictives,size(xGrid));

            subplot(2,2,mm);
            imagesc(xx,yy,gridSpace2DClusters);
            % c = colormapRedblue;
            % colormap(c);
            % colorbar;
            % caxis([0 .1])
            set(gca,'Ydir','normal');
            hold on;

        %     plot(bags.data(:,1),bags.data(:,2),'.r');
            plotTwoClusters2D(bagsDS);
            hold on;
            for k=1:K
                if (post(mm).pi(k)>myOptions.pruneThreshold)
                    if myOptions.featsUncorrelated
                        myPlotMvnEllipse(post(mm).meanN(:,k)',diag(post(mm).covN(:,k)),1,[],'b',2);
                    else
                        myPlotMvnEllipse(post(mm).meanN(:,k)',post(mm).covN(:,:,k),1,[],'b',2);
                    end
                end
        %         plotMvnEllipse(post(mm).rho(:,k)',post(mm).Lambda(:,:,k)\eye(d),1);
                % annotate contours with pi's
            end
            title(['H',num2str(MM(mm)),' Clusters (nIteration=',num2str(nIteration),')'],'FontWeight','b','FontSize',12);
            ylabel('Feature 2','FontWeight','b','FontSize',12);
            xlabel('Feature 1','FontWeight','b','FontSize',12);
            hold off;
        end
    end

    % Decision Region
    bags2DGrid.data         = cat(2,xGrid(:),yGrid(:));
    bags2DGrid.bagNum(:,1)  = 1:nGrid*nGrid;
    outputsTrain.posteriors = post;
    outputsTest = npbMilMvnTest(bags2DGrid,[],outputsTrain,myOptions);
    yOutTest    = reshape(outputsTest,size(xGrid));
    subplot(2,1,2);
    imagesc(xx,yy,yOutTest);
    set(gca,'Ydir','normal');
    hold on;
    plotTwoClusters2D(bagsDS);
    hold off;
    title('Decision Region Using NPBMIL (K=20,20)','FontSize',12);
%     axis([minXY(1), maxXY(1), minXY(2), maxXY(2)]);
    
    drawnow;
end

        
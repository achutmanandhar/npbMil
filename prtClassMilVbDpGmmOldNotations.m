classdef prtClassMilVbDpGmmOldNotations < prtClass
    % prtClassMilVbDpGmmOldNotations

    properties (SetAccess=private)
        name = 'MilVbDpGmm' % Fisher Linear Discriminant
        nameAbbreviation = 'MilVbDpGmm'            % FLD
        isNativeMary = false;  % False
    end
    
    
    properties
        rndInit = 0;
        featsUncorrelated = 0;
        plot = 0;
        pruneThreshold = .02;
        convergenceThreshold = 1e-10;
        maxIteration = 5000;
        
        K0 = 10;
        K1 = 10;
        gamma_0_1_m0
        gamma_0_2_m0
        v_0_m
        v_1_m
        beta_1_m
        beta_0_m
        rho_0_1
        rho_0_0
        Phi_0_1
        Phi_0_0
        alpha
        
        rvH1
        rvH0
        mixOfRvs
    end
    
	methods
		function self = prtClassMilVbDpGmmOldNotations(varargin)

			self = prtUtilAssignStringValuePairs(self,varargin{:});
            
            self.classTrain = 'prtDataSetClassMultipleInstance';
            self.classRun = 'prtDataSetClassMultipleInstance';
            self.classRunRetained = false;
        end
		
    end
    
    methods (Access=protected, Hidden = true)
		function self = trainAction(self,dsMil)   
            
            bags.data = dsMil.expandedData;
            bags.label = dsMil.expandedTargets;
            bags.bagNum = dsMil.bagInds;
            
            % initialize parameters
            fprintf('Initializing parameters \n');
            [prior,eStep]  = vbMilMvnInitialization(self,bags);
            vbIter = 1;
            converged  = 0;
            while converged==0   
                
                % M-step
                % Variational Posterior Approxiamtion of pi, mu, lamda (model parameters)
                post = vbMilMvnUpdate(self,bags,prior,eStep);           
                                               
                self.rvH1 = post(1).rv;
                self.rvH0 = post(2).rv;
                self.mixOfRvs = post(3).mixOfRvs;
                
                % E-step
                eStep = vbMilMvnExpectation(self,bags,eStep,post);
               
                % NFE
                nfeTerms(vbIter) = vbMilMvnNegFreeEnergy(self,prior,post,eStep);
                if (vbIter>1)
                    nfeTerms(vbIter).nfe(:,2) = nfeTerms(vbIter).nfe(:,1)-nfeTerms(vbIter-1).nfe(:,1);
                    if (nfeTerms(vbIter).nfe(:,1)<nfeTerms(vbIter-1).nfe(:,1))
                        disp(['Warning - neg free energy decreased at nIteration=',num2str(vbIter)]);
                    end
                end
                
                % Plot learned clusters
                if (self.plot)
                    nfeTermsCat = concatStructs(1,nfeTerms);
                    handle = vbMilMvnPlotLearnedClusters(self,bags,prior,post,eStep,nfeTermsCat.nfe(:,1),vbIter); % GMM model parameters (mu,Sigma) are Normal-Wishart distributed!
                end  
                
                % Check for convergence
                if vbIter>10
                    nfeTermsCat = concatStructs(1,nfeTerms);
                    [converged, ~, ~] = isconverged(nfeTermsCat.nfe(vbIter),nfeTermsCat.nfe(vbIter-1),self.convergenceThreshold);
                end
                
                if (vbIter>1)
                    if (converged==1)
                        fprintf('Convergence reached. Change in negative free energy below %.2e. \n',self.convergenceThreshold);
                    else
                        nfeTermsCat = concatStructs(1,nfeTerms);
                        fprintf('Negative Free Energy: %.2f, \t %.2e Change \n',nfeTermsCat.nfe(vbIter,1),nfeTermsCat.nfe(vbIter,2));
                    end
                end
                
                vbIter = vbIter+1;
                if (vbIter==self.maxIteration+1)
                    converged=1;
                end
                              
            end
            
        end
                
        function [prior,eStep] = vbMilMvnInitialization(self,bags)
            
            [N,D] = size(bags.data);
            KH = [self.K1 self.K0];

            % Initialize Priors
            prior(3).alpha       = [.5 .5]; % eta~Dir(alpha) = mixing proportion of the H1/H0 mixture models
        %     prior(3).alpha       = [1 9]*1000;
            for mm=1:2 % mm = mixture model
                prior(mm).gamma1 = 1;          % v|pi~Beta(gamma1,gamma2)
                prior(mm).gamma2 = 1e-3;       % sparsity inversely prop to concentration parameter
                K = KH(mm);
                prior(mm).beta = D*ones(1,K);    % mu|Gamma~N(meanMean,1/(beta*Gamma))
                prior(mm).meanMean = zeros(D,K);
                if (self.featsUncorrelated)
                    prior(mm).gamShape = ones(D,K);   % tau~Gam(gamShape,gamInvScale)
                    prior(mm).gamInvScale = 5*ones(D,K); % sparsity directly prop to inverse scale parameter
                else           
                    prior(mm).dfCov = D*ones(1,K);
                    for k=1:K
                        prior(mm).covCov(:,:,k) = (D)*eye(D); 
                    end
                end        
            end            

            % Initialize rH1H0 (big PHI)
            eStep(3).rH1H0 = zeros(N,2);              
            eStep(3).rH1H0(bags.label==1,1) = 1;
            eStep(3).rH1H0(bags.label==0,2) = 1;

            % Initialize rH (small phi)
            % rnd init         
            if (self.rndInit) 
                for mm=1:2
                    eStep(mm).rH = rand(N,KH(mm)); 
                    % Normalize
                    eStep(mm).rH = eStep(mm).rH ./ repmat(sum(eStep(mm).rH,2),1,KH(mm));
                end
            % vb init        
            else             
                % Initialize H0 respobsibility matrix (small phi)
                % Cluster instances in B- bags
                [~,ctrsH0] = kmeans(bags.data(bags.label==0,:),self.K0,'EmptyAction','singleton','MaxIter',100);
                distMatrixH0 = pdist2(bags.data,ctrsH0);
                [~,indexKH0] = min(distMatrixH0,[],2);
                eStep(2).rH = bsxfun(@eq,indexKH0,1:self.K0);   % rH0 before VB

                % Weight H0 resp (small phi) by mix mod resp (big PHI)
                eStep(2).rHrH1H0 = bsxfun(@times,eStep(2).rH,eStep(3).rH1H0(:,2));

                % Update H0 Mix Model   
                post(2) = vbMilMvnUpdateThisMixMod(self,bags,prior(2),eStep(2));         
                self.rvH0 = post(2).rv;

                % Compute p(x|H0 MM) for all instances
                eStepThisMixMod = vbMilMvnExpectationThisMixMod(self,bags,post(2),eStep(2),eStep(3).rH1H0(:,2));
                eStep(2).rH = eStepThisMixMod.rH;

%                 instancesLogPdf = npbMilLogSumLikelihood(bags,post(2),myOptions);
                instancesLogPdf = self.rvH0.logPdf(bags.data);

                % Find H1 instances
                bags.myLabelH = zeros(N,1);
                bags.myLabelH(bags.label==1 & instancesLogPdf<min(instancesLogPdf(bags.label==0)),1) = 1;

                bagNums = unique(bags.bagNum);
                totalBags = length(bagNums);
                for nBag=1:totalBags
                    if (unique(bags.label(bags.bagNum==bagNums(nBag))) == 1)
                        % enforce at least one H1 instance per positive bag
                        if ~any(bags.myLabelH(bags.bagNum==bagNums(nBag)))
                            [minVal,idx] = min(instancesLogPdf(bags.bagNum==bagNums(nBag)));
                            bags.myLabelH(bags.bagNum==bagNums(nBag) & instancesLogPdf==minVal,1) = 1;
                        end
                    end
                end

                % Initialize H1 respobsibility matrix
                % Cluster H1 instances
                [~,ctrsH1] = kmeans(bags.data(bags.myLabelH==1,:),self.K1,'EmptyAction','singleton');

                % Initialize rH1
                distMatrixH1 = pdist2(bags.data,ctrsH1);
                [~,indexKH1] = min(distMatrixH1,[],2);
                eStep(1).rH = bsxfun(@eq,indexKH1,1:self.K1);   % rH1

                % Initialize rH1H0 (big PHI)
                eStep(3).rH1H0 = zeros(N,2);
                eStep(3).rH1H0(bags.myLabelH==1,1) = 1;
                eStep(3).rH1H0(bags.myLabelH==0,2) = 1;

                % Weight small phi by big PHI
                eStep(1).rHrH1H0 = bsxfun(@times,eStep(1).rH,eStep(3).rH1H0(:,1));

                % Update H1 Mix Model
                post(1) = vbMilMvnUpdateThisMixMod(self,bags,prior(1),eStep(1));

                % Compute p(x|H0 MM) for all instances
                eStepThisMixMod = vbMilMvnExpectationThisMixMod(self,bags,post(1),eStep(1),eStep(3).rH1H0(:,1));
                eStep(1).rH = eStepThisMixMod.rH;
            end    
            % weight small phi's by big PHI
            for mm=1:2
                eStep(mm).rHrH1H0 = bsxfun(@times,eStep(mm).rH,eStep(3).rH1H0(:,mm));
            end 
   
        end
        
        function post = vbMilMvnUpdate(self,bags,prior,eStep)  
            
            for mm=1:2
                post(mm) = vbMilMvnUpdateThisMixMod(self,bags,prior(mm),eStep(mm));
            end                   
            post(3).countsH   = sum(eStep(3).rH1H0(bags.label==1,:),1);   
            post(3).countsH(post(3).countsH==0) = eps;
            post(3).alpha  = post(3).countsH + prior(3).alpha;
            post(3).eta = post(3).alpha/sum(post(3).alpha,2);
            post(3).mixOfRvs = prtRvDiscrete('probabilities',post(3).eta);
            
        end

        function post = vbMilMvnUpdateThisMixMod(self,bags,prior,eStep)    
            
            [~,D] = size(bags.data);
            K = length(prior.beta);
            resp = eStep.rHrH1H0;   
            post.counts = sum(resp,1);
            post.counts(post.counts==0) = eps;
            counts = post.counts;

            % mix
            sortObj = vbMilSortCount(counts); % If exchangeable, why sort? Why does sorting help?
            post.gamma1 = prior.gamma1 + sortObj.sortedCounts;
            post.gamma2 = prior.gamma2 + fliplr(cumsum(fliplr(sortObj.sortedCounts),2)) - sortObj.sortedCounts;
            post.pi = counts/sum(counts);

            % model
            post.beta = counts + prior.beta;
            if (self.featsUncorrelated)
                post.gamShape = repmat(counts/2,D,1) + prior.gamShape;
            else
                post.dfCov = counts + prior.dfCov;
            end    

            for k=1:K             
                % model     
                muBar(:,k) = (sum(bsxfun(@times,resp(:,k),bags.data),1)/counts(k))';
                YMinusMu   = bsxfun(@minus,bags.data,muBar(:,k)');
                if (self.featsUncorrelated==1)
                    sigmaBar(:,k) = sum(bsxfun(@times,resp(:,k),YMinusMu.^2),1) / counts(k);
                else
                    sigmaBar(:,:,k) = YMinusMu' * bsxfun(@times,resp(:,k),YMinusMu) / counts(k);
                end

                post.meanMean(:,k) = (prior.beta(k) * prior.meanMean(:,k) + counts(k)*muBar(:,k)) / post.beta(k); % eq (21)                            
                sampleMeanMinusMeanMean = muBar(:,k) - prior.meanMean(:,k);        
                if (self.featsUncorrelated==1)
                    post.gamInvScale(:,k) = prior.gamInvScale(:,k) + .5*counts(k)*sigmaBar(:,k) + ...
                        .5*(prior.beta(k)*counts(k)/(post.beta(k))) * sum(sampleMeanMinusMeanMean.^2,2); % eq (23)
                else
                    post.covCov(:,:,k) = prior.covCov(:,:,k) + counts(k)*sigmaBar(:,:,k) + ...
                        (prior.beta(k)*counts(k)/(prior.beta(k)+counts(k))) * sampleMeanMinusMeanMean * (sampleMeanMinusMeanMean'); % eq (23)
                end

                % Mean, covariance
                post.meanN(:,k) = post.meanMean(:,k)'; 
                if (self.featsUncorrelated==1)
                    post.covN(:,:,k) = diag(post.gamInvScale(:,k)./post.gamShape(:,k)); 
                else
                    post.covN(:,:,k) = post.covCov(:,:,k)/post.dfCov(k);
                  
                    if ~(isCovPosSemiDef(post.covN(:,:,k)))
                        post.covN(:,:,k) = roundn(post.covN(:,:,k),-4);
                    end
                end       

                % Find parameters of the Student t-distribution
                if (self.featsUncorrelated==1)
                    post.degreesOfFreedom(k) = post.gamShape(k) + 1 - D;
                    post.Lambda(:,:,k) = ( (post.beta(k) + 1)/(post.beta(k)*post.degreesOfFreedom(k)) ) * post.gamInvScale(:,k); 
                else
                    post.degreesOfFreedom(k) = post.dfCov(k) + 1 - D;
                    post.Lambda(:,:,k) = ( (post.beta(k)*post.degreesOfFreedom(k))/(post.beta(k) + 1) ) * (post.covCov(:,:,k)\eye(D));
                end        
                mixMod(k) = prtRvMvn('mu',post.meanMean(:,k),'sigma',post.covN(:,:,k));
            end                 
            post.rv = prtRvGmm('nComponents',K,'mixingProportions',post.pi,'components',mixMod);
            
        end
        
        function eStep = vbMilMvnExpectation(self,bags,eStep,post) 
            
            eStep(3).logEtaTilda    = psi(0,post(3).alpha) - psi(0,sum(post(3).alpha));
            for mm=1:2
                eStepThisMixMod = vbMilMvnExpectationThisMixMod(self,bags,post(mm),eStep(mm),eStep(3).rH1H0(:,mm));
                eStep(mm).rH = eStepThisMixMod.rH;
                eStep(mm).rHrH1H0 = eStepThisMixMod.rHrH1H0;
                eStep(mm).variationalAvgLL = eStepThisMixMod.variationalAvgLL;
                eStep(mm).logPiTilda = eStepThisMixMod.logPiTilda;
                eStep(3).logrH1H0Tilda(:,mm) = sum(eStep(mm).rH .* eStep(mm).variationalAvgLL,2) + eStep(3).logEtaTilda(mm); % [N*1]
            end    

            % Normalize
            eStep(3).rH1H0 = exp(bsxfun(@minus,eStep(3).logrH1H0Tilda,prtUtilSumExp(eStep(3).logrH1H0Tilda')')); % eq (15)

                % If a sample is not resposible to anyone randomly assign him
                % Ideally if initializations were good this would never happen. 
                noResponsibilityPoints = find(sum(eStep(3).rH1H0,2)==0);
                if ~isempty(noResponsibilityPoints)
                    for iNrp = 1:length(noResponsibilityPoints)
                        eStep(3).rH1H0(noResponsibilityPoints(iNrp),:) = .5;
                    end
                end

            if any(isnan(eStep(mm).rH1H0))
                keyboard;
            end

            % enforce correct responsibilities for instances in B- bags
            eStep(3).rH1H0(bags.label==0,1) = 0;
            eStep(3).rH1H0(bags.label==0,2) = 1;       

%                 % Hack!
%                 % force at least one H1 instance per positive bag
%                 if myOptions.forceOneH1InstancePerPosBag
%                     bagNums = unique(bags.bagNum);
%                     totalBags = length(bagNums);
%                     for nBag=1:totalBags
%                         isThisBag = bags.bagNum==bagNums(nBag);
%                         thisBagIndices = find(isThisBag);
%                         if unique(bags.label(thisBagIndices))==1
%                             if ~any(eStep(3).rH1H0(thisBagIndices,1))
%                                 % enforce
%                                 [~,idx] = max(mixModLogLikelihoods(thisBagIndices,1));
%                                 eStep(3).rH1H0(thisBagIndices(idx),1) = 1;
%                                 eStep(3).rH1H0(thisBagIndices(idx),2) = 0;
%                             end
%                         end
%                     end
%                 end

            % weight small phi's by big PHI
            for mm=1:2
                eStep(mm).rHrH1H0 = bsxfun(@times,eStep(mm).rH,eStep(3).rH1H0(:,mm));
            end
            
        end

        function eStep = vbMilMvnExpectationThisMixMod(self,bags,post,eStep,rH1H0ThisMixMod)         
            
            [N,D] = size(bags.data);
            K = length(post.pi);
            eStep.logPiTilda        = nan(1,K);
            eStep.logGammaTilda     = nan(1,K);
            eStep.variationalAvgLL  = nan(N,K);
            eStep.logrHTilda        = nan(N,K);

            % E-Step    
            % mix
            eStep.logVTildaSorted            = psi(0,post.gamma1) - psi(0,post.gamma1+post.gamma2);
            eStep.log1MinusVTildaSorted      = psi(0,post.gamma2) - psi(0,post.gamma1+post.gamma2);
            for k=1:K
                if (k==1)
                    eStep.logPiTildaSorted(k)        = eStep.logVTildaSorted(k);
                else
                    eStep.logPiTildaSorted(k)        = eStep.logVTildaSorted(k) + sum(eStep.log1MinusVTildaSorted(1:k-1),2);
                end
            end    
            % Unsort
            sortObj = vbDpGmmSortCount(post.counts);
            eStep.logPiTilda = eStep.logPiTildaSorted(sortObj.unsortIndices);

            dims = 1:D;
            % model
            for k=1:K        
                YMinusmeanMean           = bsxfun(@minus,bags.data,post.meanMean(:,k)');      
                if (self.featsUncorrelated)
                    eStep.logTauTilda(:,k)    = psi(0,post.gamShape(:,k)/2) - log(post.gamInvScale(:,k)); 
                    expectMahalDist     = YMinusmeanMean.^2 * (post.gamShape(:,k)./post.gamInvScale(:,k)) + 1/post.beta(k);
            %             expectMahalDist     = sum(bsxfun(@times,(post.gamShape(:,k)./post.gamInvScale(:,k))',YMinusmeanMean.^2) + 1/post.beta(k),2);
                    eStep.variationalAvgLL(:,k)     = -0.5*D*log(2*pi) + 0.5*sum(eStep.logTauTilda(:,k),1) - expectMahalDist;
                else
                    eStep.logGammaTilda(:,k)      = sum( psi(0,(post.dfCov(:,k)+1-dims)/2) ) - prtUtilLogDet(post.covCov(:,:,k)) + D*log(2);
                    eStep.GammaBarInv(:,:,k)    = post.covCov(:,:,k)/post.dfCov(k) + 1e-4; % 1e-4 for regularization
                        if ~(isCovPosSemiDef(eStep.GammaBarInv(:,:,k)))
                            eStep.GammaBarInv(:,:,k) = roundn(eStep.GammaBarInv(:,:,k),-4);
                        end
                    expectMahalDist  = prtUtilCalcDiagXcInvXT(YMinusmeanMean,eStep.GammaBarInv(:,:,k)) + D/post.beta(k);
                        if (any(expectMahalDist<0)) % distance can't be negative
                            disp('Error - distance cannot be negative (set to eps)');
                            keyboard;
                %                 expectMahalDist(expectMahalDist(:,k)<0,k)=eps;
                        end
                    eStep.variationalAvgLL(:,k)     = 0.5*(-D*log(2*pi) + eStep.logGammaTilda(k) - expectMahalDist);
                end
            %         if any((eStep.variationalAvgLL(:,k)==-inf|eStep.variationalAvgLL(:,k)==0))
            %             keyboard;
            % %             eStep.variationalAvgLL(eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)=min(eStep.variationalAvgLL(eStep.variationalAvgLL~=0));
            %         end
            %         eStep.logrHTilda(:,k)   = rH1H0ThisMixMod.*eStep.variationalAvgLL(:,k) + eStep.logPiTilda(k);
            end

            if any(any((eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)))
                keyboard;
                eStep.variationalAvgLL(eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)=min(eStep.variationalAvgLL(eStep.variationalAvgLL~=0));
            end

            eStep.logrHTilda = bsxfun( @plus,eStep.logPiTilda,bsxfun(@times,rH1H0ThisMixMod,eStep.variationalAvgLL) );  % sim to eq (13) % [N*1]

            if ( any(isinf(eStep.logrHTilda(:,k))) || any(isnan(eStep.logrHTilda(:,k))) ) % check for inf/nan
                disp('Error - inf/nan in logResp values');
                keyboard;
            end

            % Normalize
            eStep.rH = exp(bsxfun(@minus,eStep.logrHTilda,prtUtilSumExp(eStep.logrHTilda')'));

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
            
        end
        
        function nfeTerms = vbMilMvnNegFreeEnergy(self,prior,post,eStep)
             
            KLz = zeros(1,2);
            weightedVariationalAvgLL = zeros(1,2);
            for mm=1:2
                weightedVariationalAvgLL(mm) = sum(sum(eStep(mm).rHrH1H0 .* eStep(mm).variationalAvgLL,1),2);

                respSmallzInfoTerm = eStep(mm).rHrH1H0.*log(eStep(mm).rH);
                respSmallzInfoTerm(eStep(mm).rHrH1H0==0) = 0;

                expectLogPiTerm = post(mm).counts.*eStep(mm).logPiTilda;
                KLz(mm) = sum( (sum(respSmallzInfoTerm,1) - expectLogPiTerm), 2);
%                 KLz(mm) = sum( sum(respSmallzInfoTerm,1) - suffStat(mm).counts.*eStep(mm).logPiTilda, 2);
            end

            respOfMixModInfoTerm = eStep(3).rH1H0.*log(eStep(3).rH1H0);
            respOfMixModInfoTerm(eStep(3).rH1H0==0)=0;
            KLZeta = sum(respOfMixModInfoTerm) - post(3).countsH.*eStep(3).logEtaTilda;

            sumWeightedVariationalAvgLL = sum(weightedVariationalAvgLL);
            sumKLZeta = -sum(KLZeta);
            sumKLz = -sum(KLz);
            
            sumKLeta = KLDirichlet(post(3).alpha,prior(3).alpha);

            mthKLv = zeros(1,2);
            mthKLMuAndGamma = zeros(1,2);
            for mm=1:2 % mm = nth mixture model
                K = length(post(mm).pi);
                KLv       = zeros(1,K);
                KLMuAndGamma = zeros(1,K);
                for k=1:K
                    KLv(k) = KLBeta(post(mm).gamma1(k),post(mm).gamma2(k),prior(mm).gamma1,prior(mm).gamma2);
                    if (self.featsUncorrelated)
                        KLMuAndGamma(k) = KLNormalGamma(post(mm).meanMean(:,k),prior(mm).meanMean(:,k),post(mm).beta(k),prior(mm).beta(k),post(mm).gamShape(:,k),prior(mm).gamShape(:,k),post(mm).gamInvScale(:,k),prior(mm).gamInvScale(:,k));
                    else               
                        KLMuAndGamma(k) = KLNormalWishart(post(mm).meanMean(:,k),prior(mm).meanMean(:,k),post(mm).beta(k),prior(mm).beta(k),post(mm).dfCov(k),prior(mm).dfCov(k),post(mm).covCov(:,:,k),prior(mm).covCov(:,:,k));
                    end            
                end
                mthKLv(mm) = sum(KLv);
                mthKLMuAndGamma(mm) = sum(KLMuAndGamma);
            end

            sumKLv = sum(mthKLv);
            sumKLMuAndGamma = sum(mthKLMuAndGamma);
            
            F = sumWeightedVariationalAvgLL + sumKLZeta - sumKLeta + sumKLz - sumKLv - sumKLMuAndGamma;

            if (isinf(F))
                disp('Error! Neg Free Nergy is INF!');
            %     keyboard;
            end

            nfeTerms.nfe        = zeros(1,2);
            nfeTerms.nfe(:,1)   = F;
            nfeTerms.Lav        = sumWeightedVariationalAvgLL;
            nfeTerms.KLZeta     = sumKLZeta;
            nfeTerms.KLz        = sumKLz;
            nfeTerms.KLeta      = sumKLeta;
            nfeTerms.KLv        = sumKLv;
            nfeTerms.KLMuAndGamma = sumKLMuAndGamma;            
            
        end        
        
        function sortObj = vbMilSortCount(counts)

            sortObj.unsortedCounts = counts;
            [sortObj.sortedCounts, sortObj.sortIndices] = sort(counts,2,'descend');
            [~, sortObj.unsortIndices] = sort(sortObj.sortIndices,2,'ascend');

        end        
        
        function fig1 = vbMilMvnPlotLearnedClusters(self,bags,prior,post,eStep,negFreeEnergy,nIteration)
            
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
                        if (post(mm).pi(k)>self.pruneThreshold)
                            if self.featsUncorrelated              
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

                    subplot(2,1,mm);
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
                        if (post(mm).pi(k)>self.pruneThreshold)
                            if self.featsUncorrelated
                                plotMvnEllipse(post(mm).meanN(:,k)',diag(post(mm).covN(:,k)),1);
                            else
                                plotMvnEllipse(post(mm).meanN(:,k)',post(mm).covN(:,:,k),1);
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
            subplot(1,3,1);
            Kmax = max([length(post(1).pi) length(post(2).pi)]);
            for mm=1:2
                stem(post(mm).pi,'fill','MarkerFaceColor',colorVector(mm),'Color',colorVector(mm),'LineWidth',2); grid on; grid minor;
                axis([1 Kmax+1 0 1]);
                hold on;
            end
            ylabel('\pi_i','FontWeight','b','FontSize',14);
            xlabel('components in a mixture model','FontWeight','b','FontSize',12);
            title(['nIteration=',num2str(nIteration)],'FontWeight','b','FontSize',12);
            legend('{\pi_i}^{1}','{\pi_i}^{0}','location','best');
            hold off;

            subplot(1,3,2);
            stem(post(3).eta,'fill','MarkerFaceColor',colorVector(mm),'Color','k','LineWidth',2); grid on;
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
            % imagesc(eStep(3).rH1H0); 
            % title(['\phi^{M}_{lm}'],'FontWeight','b');%colorbar;
            % ylabel('instances','FontWeight','b');
            % xlabel('m=\{1,0\}','FontWeight','b');
            % 
            subplot(1,3,3);
            plot(negFreeEnergy,'.'); grid on;
            ylabel('Neg Free Energy','FontWeight','b');
            xlabel('nIteration','FontWeight','b');
            
        end
        
        function yOut = runAction(self,dsMil)      
            
            for n = 1:dsMil.nObservations            
                milStruct = dsMil.data(n);
                data = milStruct.data; 
                
                h1 = self.rvH1.logPdf(data);
                h0 = self.rvH0.logPdf(data);           
                
                % Eqn 46
                h1Weighted = h1 + log(self.mixOfRvs.probabilities(1)); 
                h0Weighted = h0 + log(self.mixOfRvs.probabilities(2)); 
                
                instancesInThisBagLLGivenPosBag = prtUtilSumExp(cat(2,h1Weighted,h0Weighted)')';
                thisBagLLGivenPosBag = sum(instancesInThisBagLLGivenPosBag,1);
                thisBagLLGivenNegBag = sum(h0,1);
                
                bagLL = [thisBagLLGivenPosBag,thisBagLLGivenNegBag] - prtUtilSumExp([thisBagLLGivenPosBag,thisBagLLGivenNegBag]')';
                y(n,1) = bagLL(:,1)-bagLL(:,2);

            end
            
            yOut = prtDataSetClass(y,dsMil.targets);
        end
        
    end
end

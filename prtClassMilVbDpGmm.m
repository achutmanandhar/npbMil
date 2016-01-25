classdef prtClassMilVbDpGmm < prtClass
    % prtClassMilVbDpGmm

    properties (SetAccess=private)
        name = 'MilVbDpGmm' % Varitional Bayes Dirichlet Process Gaussian Mixture Model
        nameAbbreviation = 'MilVbDpGmm'  % Varitional Bayes Dirichlet Process Gaussian Mixture Model
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
		function self = prtClassMilVbDpGmm(varargin)

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
                % Variational Posterior Approximation of pi, mu, lamda (model parameters)
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
            prior(3).alpha_m       = [.5 .5]; % eta_m~Dir(alpha_m) = mixing proportion of the H1/H0 mixture models
            for mm=1:2 % mm = mixture model
                % Priors in Eqn 7
                prior(mm).gamma_0_1_m = 1;          
                prior(mm).gamma_0_2_m = 1e-3; % sparsity inversely prop to concentration parameter
                                              % lower values = sparser model, Beta(1,1e-3) ~ 1, Beta(1,1) = U(0,1), Beta(1,1e+3) ~ 0 
                K_m = KH(mm);
                % Priors in Eqn 12, 13, 14
                prior(mm).beta_0_m = D;    
                prior(mm).rho_0_m = zeros(D,K_m);
                if (self.featsUncorrelated)
                    prior(mm).del_0_1_m = 1;   
                    prior(mm).del_0_2_m = 5; % sparsity directly prop to inverse scale parameter
                else           
                    prior(mm).nu_0_m = D;
                    for k=1:K_m
                        prior(mm).Phi_0_m(:,:,k) = (D)*eye(D); 
                    end
                end        
            end            

            % Initialize phi_l_m_M 
            eStep(3).phi_l_m_M = zeros(N,2);              
            eStep(3).phi_l_m_M(bags.label==1,1) = 1;
            eStep(3).phi_l_m_M(bags.label==0,2) = 1;

            % Initialize phi_l_i_m
            % random initialization         
            if (self.rndInit) 
                for mm=1:2
                    eStep(mm).phi_l_i_m = rand(N,KH(mm)); 
                    % Normalize
                    eStep(mm).phi_l_i_m = eStep(mm).phi_l_i_m ./ repmat(sum(eStep(mm).phi_l_i_m,2),1,KH(mm));
                end
            % VB initialziation        
            else             
                % Initialize H0 respobsibility matrix (phi_l_i_0)
                % Cluster instances in B- bags
                [~,ctrsH0] = kmeans(bags.data(bags.label==0,:),self.K0,'EmptyAction','singleton','MaxIter',100);
                distMatrixH0 = pdist2(bags.data,ctrsH0);
                [~,indexKH0] = min(distMatrixH0,[],2);
                eStep(2).phi_l_i_m = bsxfun(@eq,indexKH0,1:self.K0);   % phi_l_i_0 before VB

                % Weight phi_l_i_0 by phi_l_0_M
                eStep(2).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(2).phi_l_i_m,eStep(3).phi_l_m_M(:,2));

                % Update H0 Mix Model (M-step for only H0 mixture model)   
                post(2) = vbMilMvnUpdateThisMixMod(self,bags,prior(2),eStep(2));         
                self.rvH0 = post(2).rv;
                
                % E-step only for H1 mixture model
                eStepThisMixMod = vbMilMvnExpectationThisMixMod(self,bags,post(2),eStep(2),eStep(3).phi_l_m_M(:,2));
                eStep(2).phi_l_i_m = eStepThisMixMod.phi_l_i_m;

                % Compute log p(x_l|H0 MM); l={1,2,...,N_T}
                instancesLogPdf = self.rvH0.logPdf(bags.data);

                % Find most likely H1 instances in positive bags
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

                % Initialize H1 respobsibility matrix (phi_l_i_1)
                % Cluster H1 instances
                [~,ctrsH1] = kmeans(bags.data(bags.myLabelH==1,:),self.K1,'EmptyAction','singleton');

                % Initialize phi_l_i_1
                distMatrixH1 = pdist2(bags.data,ctrsH1);
                [~,indexKH1] = min(distMatrixH1,[],2);
                eStep(1).phi_l_i_m = bsxfun(@eq,indexKH1,1:self.K1);   % phi_l_i_1

                % Initialize phi_l_m_M
                eStep(3).phi_l_m_M = zeros(N,2);
                eStep(3).phi_l_m_M(bags.myLabelH==1,1) = 1;
                eStep(3).phi_l_m_M(bags.myLabelH==0,2) = 1;

                % Weight phi_l_i_0 by phi_l_0_M
                eStep(1).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(1).phi_l_i_m,eStep(3).phi_l_m_M(:,1));

                % Update H1 Mix Model (M-step only for H1 mixture model)
                post(1) = vbMilMvnUpdateThisMixMod(self,bags,prior(1),eStep(1));

                % E-step only for H1 mixture model
                eStepThisMixMod = vbMilMvnExpectationThisMixMod(self,bags,post(1),eStep(1),eStep(3).phi_l_m_M(:,1));
                eStep(1).phi_l_i_m = eStepThisMixMod.phi_l_i_m;
            end    
            % weight small phi's by big PHI
            for mm=1:2
                eStep(mm).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(mm).phi_l_i_m,eStep(3).phi_l_m_M(:,mm));
            end 
   
        end
        
        function post = vbMilMvnUpdate(self,bags,prior,eStep)  
            
            for mm=1:2
                % M-step of the mth mixture model
                post(mm) = vbMilMvnUpdateThisMixMod(self,bags,prior(mm),eStep(mm));
            end                   
            % Eqn 24
            post(3).N_m_bar   = sum(eStep(3).phi_l_m_M(bags.label==1,:),1);   
            post(3).N_m_bar(post(3).N_m_bar==0) = eps;
            post(3).alpha_m  = post(3).N_m_bar + prior(3).alpha_m;
            post(3).eta_m = post(3).alpha_m/sum(post(3).alpha_m,2);
            post(3).mixOfRvs = prtRvDiscrete('probabilities',post(3).eta_m);
            
        end

        function post = vbMilMvnUpdateThisMixMod(self,bags,prior,eStep)    
            
            [~,D] = size(bags.data);

            K_m = size(prior.rho_0_m,2);
            resp = eStep.phi_l_m_M_phi_l_i_m;
            % Eqn 25
            post.N_i_m_bar = sum(resp,1);
            post.N_i_m_bar(post.N_i_m_bar==0) = eps;
            N_i_m_bar = post.N_i_m_bar;

            % mix
            sortObj = vbMilSortCount(N_i_m_bar);             
            % Eqn 20
            post.gamma_i_1_m = prior.gamma_0_1_m + sortObj.sortedCounts;
            % Eqn 21
            post.gamma_i_2_m = prior.gamma_0_2_m + fliplr(cumsum(fliplr(sortObj.sortedCounts),2)) - sortObj.sortedCounts;
            post.pi_i_m = N_i_m_bar/sum(N_i_m_bar);

            % model
            % Eqn 32 or 38
            post.beta_i_m = N_i_m_bar + prior.beta_0_m;
            if (self.featsUncorrelated)
                % Eqn 40
                post.del_i_1_m = repmat(N_i_m_bar/2,D,1) + prior.del_0_1_m;
            else
                % Eqn 31
                post.nu_i_m = N_i_m_bar + prior.nu_0_m;
            end    

            for k=1:K_m             
                % model
                % Eqn 35 or Eqn 42
                mu_i_m_bar(:,k) = (sum(bsxfun(@times,resp(:,k),bags.data),1)/N_i_m_bar(k))';
                x_l_minus_mu_i_m_bar   = bsxfun(@minus,bags.data,mu_i_m_bar(:,k)');
                if (self.featsUncorrelated==1)
                    % Eqn 43
                    sigma_i_m_bar(:,k) = sum(bsxfun(@times,resp(:,k),x_l_minus_mu_i_m_bar.^2),1) / N_i_m_bar(k);
                else
                    % Eqn 36
                    sigma_i_m_bar(:,:,k) = x_l_minus_mu_i_m_bar' * bsxfun(@times,resp(:,k),x_l_minus_mu_i_m_bar) / N_i_m_bar(k);
                end
                % Eqn 33 or Eqn 39
                post.rho_i_m(:,k) = (prior.beta_0_m * prior.rho_0_m(:,k) + N_i_m_bar(k)*mu_i_m_bar(:,k)) / post.beta_i_m(k); % eq (21)                            
                sampleMeanMinusMeanMean = mu_i_m_bar(:,k) - prior.rho_0_m(:,k);        
                if (self.featsUncorrelated==1)
                    % Eqn 41
                    post.del_i_2_m(:,k) = prior.del_0_2_m + .5*N_i_m_bar(k)*sigma_i_m_bar(:,k) + ...
                        .5*(prior.beta_0_m*N_i_m_bar(k)/(post.beta_i_m(k))) * sum(sampleMeanMinusMeanMean.^2,2); % eq (23)
                else
                    % Eqn 34
                    post.Phi_i_m(:,:,k) = prior.Phi_0_m(:,:,k) + N_i_m_bar(k)*sigma_i_m_bar(:,:,k) + ...
                        (prior.beta_0_m*N_i_m_bar(k)/(prior.beta_0_m+N_i_m_bar(k))) * sampleMeanMinusMeanMean * (sampleMeanMinusMeanMean'); % eq (23)
                end

                % Expected posterior mean
                post.mu_i_m(:,k) = post.rho_i_m(:,k)'; 
                if (self.featsUncorrelated==1)
                    % Expected the posterior covariance
                    post.GammaInv_i_m(:,:,k) = diag(post.del_i_2_m(:,k)./post.del_i_1_m(:,k)); 
                else
                    % Expected posterior covariance
                    post.GammaInv_i_m(:,:,k) = post.Phi_i_m(:,:,k)/post.nu_i_m(k);
                                        
                    if ~(isCovPosSemiDef(post.GammaInv_i_m(:,:,k)))
                        post.GammaInv_i_m(:,:,k) = roundn(post.GammaInv_i_m(:,:,k),-4);
                    end
                end       

                % Find parameters of the Student t-distribution
                if (self.featsUncorrelated==1)
                    post.degreesOfFreedom(k) = post.del_i_1_m(k) + 1 - D;
                    post.Lambda(:,:,k) = ( (post.beta_i_m(k) + 1)/(post.beta_i_m(k)*post.degreesOfFreedom(k)) ) * post.del_i_2_m(:,k); 
                else
                    % Eqn 135
                    post.degreesOfFreedom(k) = post.nu_i_m(k) + 1 - D;
                    % Eqn 137
                    post.Lambda(:,:,k) = ( (post.beta_i_m(k)*post.degreesOfFreedom(k))/(post.beta_i_m(k) + 1) ) * (post.Phi_i_m(:,:,k)\eye(D));
                end        
                mixMod(k) = prtRvMvn('mu',post.rho_i_m(:,k),'sigma',post.GammaInv_i_m(:,:,k));
            end                 
            post.rv = prtRvGmm('nComponents',K_m,'mixingProportions',post.pi_i_m,'components',mixMod);
            
        end
        
        function eStep = vbMilMvnExpectation(self,bags,eStep,post) 
            % Eqn 108
            eStep(3).expect_log_eta_m    = psi(0,post(3).alpha_m) - psi(0,sum(post(3).alpha_m));
            for mm=1:2
                % E-step of the mth mixture model
                eStepThisMixMod = vbMilMvnExpectationThisMixMod(self,bags,post(mm),eStep(mm),eStep(3).phi_l_m_M(:,mm));
                eStep(mm).phi_l_i_m = eStepThisMixMod.phi_l_i_m;
%                 eStep(mm).phi_l_m_M_phi_l_i_m =
%                 eStepThisMixMod.phi_l_m_M_phi_l_i_m; % Modified
%                 10/17/2012 AM
                eStep(mm).variationalAvgLL = eStepThisMixMod.variationalAvgLL;
                eStep(mm).expect_log_pi_i_m = eStepThisMixMod.expect_log_pi_i_m;
                % Eqn 58
                % Modified AM May 6, 2013
%                 eStep(3).log_phi_l_m_M_hat(:,mm) = sum(eStep(mm).variationalAvgLL,2) + eStep(3).expect_log_eta_m(mm); % [N*1]
                eStep(3).log_phi_l_m_M_hat(:,mm) = sum(eStep(mm).phi_l_i_m .* eStep(mm).variationalAvgLL,2) + eStep(3).expect_log_eta_m(mm); % [N*1]                
            end    

            % Normalize Eqn 59
            eStep(3).phi_l_m_M = exp(bsxfun(@minus,eStep(3).log_phi_l_m_M_hat,prtUtilSumExp(eStep(3).log_phi_l_m_M_hat')')); % eq (15)

                % If a sample is not resposible to anyone randomly assign him
                % Ideally if initializations were good this would never happen. 
                noResponsibilityPoints = find(sum(eStep(3).phi_l_m_M,2)==0);
                if ~isempty(noResponsibilityPoints)
                    for iNrp = 1:length(noResponsibilityPoints)
                        eStep(3).phi_l_m_M(noResponsibilityPoints(iNrp),:) = .5;
                    end
                end

            if any(isnan(eStep(3).phi_l_m_M))
                keyboard;
            end

            % enforce correct responsibilities for instances in B- bags
            eStep(3).phi_l_m_M(bags.label==0,1) = 0;
            eStep(3).phi_l_m_M(bags.label==0,2) = 1;       

            % weight small phi's by big PHI
            for mm=1:2
                eStep(mm).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(mm).phi_l_i_m,eStep(3).phi_l_m_M(:,mm));
            end
            
        end

        function eStep = vbMilMvnExpectationThisMixMod(self,bags,post,eStep,rH1H0ThisMixMod)         
            
            [N,D] = size(bags.data);
            K_m = length(post.pi_i_m);
            eStep.expect_log_pi_i_m = nan(1,K_m);
            eStep.expect_logdet_Gamma_i_m = nan(1,K_m);
            eStep.variationalAvgLL = nan(N,K_m);
            eStep.log_phi_l_i_m_hat = nan(N,K_m);
            
            % mix
            % Eqn 106
            eStep.expect_log_v_i_m_sorted = psi(0,post.gamma_i_1_m) - psi(0,post.gamma_i_1_m+post.gamma_i_2_m);
            % Eqn 107
            eStep.expect_log_1_minus_v_i_m_sorted = psi(0,post.gamma_i_2_m) - psi(0,post.gamma_i_1_m+post.gamma_i_2_m);            
            % Eqn 105
            for k=1:K_m
                if (k==1)
                    eStep.expect_log_pi_i_m_sorted(k) = eStep.expect_log_v_i_m_sorted(k);
                else
                    eStep.expect_log_pi_i_m_sorted(k) = eStep.expect_log_v_i_m_sorted(k) + sum(eStep.expect_log_1_minus_v_i_m_sorted(1:k-1),2);
                end
            end    
            % Unsort
            sortObj = vbMilSortCount(post.N_i_m_bar);            
            eStep.expect_log_pi_i_m = eStep.expect_log_pi_i_m_sorted(sortObj.unsortIndices);

            dims = 1:D;
            % model
            for k=1:K_m        
                x_l_minus_rho_i_m = bsxfun(@minus,bags.data,post.rho_i_m(:,k)');      
                if (self.featsUncorrelated)
                    % Eqn 111
                    eStep.expect_logdet_tau_i_m(:,k) = psi(0,post.del_i_1_m(:,k)/2) - log(post.del_i_2_m(:,k)); 
                    % Eqn 112
                    expectMahalDist     = x_l_minus_rho_i_m.^2 * (post.del_i_1_m(:,k)./post.del_i_2_m(:,k)) + 1/post.beta_i_m(k);     
                    % Eqn 118
                    eStep.variationalAvgLL(:,k) = -0.5*D*log(2*pi) + 0.5*sum(eStep.expect_logdet_tau_i_m(:,k),1) - expectMahalDist;
                else
                    % Eqn 109
                    eStep.expect_logdet_Gamma_i_m(:,k) = sum( psi(0,(post.nu_i_m(:,k)+1-dims)/2) ) - prtUtilLogDet(post.Phi_i_m(:,:,k)) + D*log(2);
                    eStep.GammaBarInv(:,:,k)    = post.Phi_i_m(:,:,k)/post.nu_i_m(k) + 1e-4; % 1e-4 for regularization
                        if ~(isCovPosSemiDef(eStep.GammaBarInv(:,:,k)))
                            eStep.GammaBarInv(:,:,k) = roundn(eStep.GammaBarInv(:,:,k),-4);
                        end
                    % Eqn 110
                    expectMahalDist  = prtUtilCalcDiagXcInvXT(x_l_minus_rho_i_m,eStep.GammaBarInv(:,:,k)) + D/post.beta_i_m(k);
                        if (any(expectMahalDist<0)) % distance can't be negative
                            disp('Error - distance cannot be negative (set to eps)');
                            % keyboard;
                            % expectMahalDist(expectMahalDist(:,k)<0,k)=eps;
                        end
                    % Eqn 116
                    eStep.variationalAvgLL(:,k)     = 0.5*(-D*log(2*pi) + eStep.expect_logdet_Gamma_i_m(k) - expectMahalDist);
                end
            end

            if any(any((eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)))
                keyboard;
                eStep.variationalAvgLL(eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)=min(eStep.variationalAvgLL(eStep.variationalAvgLL~=0));
            end

            % Eqn 53      
            % Modified AM May 6, 2013
%             eStep.log_phi_l_i_m_hat = bsxfun( @plus,eStep.expect_log_pi_i_m,eStep.variationalAvgLL );  % sim to eq (13) % [N*1]
            eStep.log_phi_l_i_m_hat = bsxfun( @plus,eStep.expect_log_pi_i_m,bsxfun(@times,rH1H0ThisMixMod,eStep.variationalAvgLL) );  % sim to eq (13) % [N*1]

            if ( any(isinf(eStep.log_phi_l_i_m_hat(:,k))) || any(isnan(eStep.log_phi_l_i_m_hat(:,k))) ) % check for inf/nan
                disp('Error - inf/nan in logResp values');
                keyboard;
            end

            % Normalize (Eqn 54)
            eStep.phi_l_i_m = exp(bsxfun(@minus,eStep.log_phi_l_i_m_hat,prtUtilSumExp(eStep.log_phi_l_i_m_hat')'));

            % If a sample is not resposible to anyone randomly assign him
            % Ideally if initializations were good this would never happen. 
            noResponsibilityPoints = find(sum(eStep.phi_l_i_m,2)==0);
            if ~isempty(noResponsibilityPoints)
                for iNrp = 1:length(noResponsibilityPoints)
                    eStep.phi_l_i_m(noResponsibilityPoints(iNrp),:) = 1/K_m;
                end
            end
            if any(isnan(eStep.phi_l_i_m))
                keyboard;
            end
            
        end
        
        function nfeTerms = vbMilMvnNegFreeEnergy(self,prior,post,eStep)
             
            KLz = zeros(1,2);
            weightedVariationalAvgLL = zeros(1,2);
            for mm=1:2
                weightedVariationalAvgLL(mm) = sum(sum(eStep(mm).phi_l_m_M_phi_l_i_m .* eStep(mm).variationalAvgLL,1),2);

                respSmallzInfoTerm = eStep(mm).phi_l_m_M_phi_l_i_m.*log(eStep(mm).phi_l_i_m);
                respSmallzInfoTerm(eStep(mm).phi_l_m_M_phi_l_i_m==0) = 0;

                expectLogPiTerm = post(mm).N_i_m_bar.*eStep(mm).expect_log_pi_i_m;
                % Eqn 123
                KLz(mm) = sum( (sum(respSmallzInfoTerm,1) - expectLogPiTerm), 2);
%                 KLz(mm) = sum( sum(respSmallzInfoTerm,1) - post(mm).N_i_m_bar.*eStep(mm).logPiTilda, 2);
            end

            respOfMixModInfoTerm = eStep(3).phi_l_m_M.*log(eStep(3).phi_l_m_M);
            respOfMixModInfoTerm(eStep(3).phi_l_m_M==0)=0;
            % Eqn 126
            KLZeta = sum(respOfMixModInfoTerm) - post(3).N_m_bar.*eStep(3).expect_log_eta_m;

            sumWeightedVariationalAvgLL = sum(weightedVariationalAvgLL);
            sumKLZeta = -sum(KLZeta);
            sumKLz = -sum(KLz);
            
            sumKLeta = KLDirichlet(post(3).alpha_m,prior(3).alpha_m);

            mthKLv = zeros(1,2);
            mthKLMuAndGamma = zeros(1,2);
            for mm=1:2 % mm = mth mixture model
                K = length(post(mm).pi_i_m);
                KLv       = zeros(1,K);
                KLMuAndGamma = zeros(1,K);
                for k=1:K
                    KLv(k) = KLBeta(post(mm).gamma_i_1_m(k),post(mm).gamma_i_2_m(k),prior(mm).gamma_0_1_m,prior(mm).gamma_0_2_m);
                    if (self.featsUncorrelated)
                        % Eqn 128
                        KLMuAndGamma(k) = KLNormalGamma(post(mm).rho_i_m(:,k),prior(mm).rho_0_m(:,k),post(mm).beta_i_m(k),prior(mm).beta_0_m,post(mm).del_i_1_m(:,k),prior(mm).del_0_1_m,post(mm).del_i_2_m(:,k),prior(mm).del_0_2_m);
                    else
                        % Eqn 127
                        KLMuAndGamma(k) = KLNormalWishart(post(mm).rho_i_m(:,k),prior(mm).rho_0_m(:,k),post(mm).beta_i_m(k),prior(mm).beta_0_m,post(mm).nu_i_m(k),prior(mm).nu_0_m,post(mm).Phi_i_m(:,:,k),prior(mm).Phi_0_m(:,:,k));
                    end            
                end
                mthKLv(mm) = sum(KLv);
                mthKLMuAndGamma(mm) = sum(KLMuAndGamma);
            end

            sumKLv = sum(mthKLv);
            sumKLMuAndGamma = sum(mthKLMuAndGamma);
            
            % Eqn 114
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
        
        function fig1 = vbMilMvnPlotLearnedClusters(self,bags,prior,post,eStep,negFreeEnergy,nIteration)
            
            fig1 = figure(10);
            d = size(bags.data,2);
            colorVector = ['r','b'];
            
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
                    K = length(post(mm).pi_i_m);
                    for k=1:K
                        if (post(mm).pi_i_m(k)>self.pruneThreshold)
                            if self.featsUncorrelated              
                                predictives = predictives + post(mm).pi_i_m(k)*mvnpdf(gridSpace2D,post(mm).mu_i_m(:,k)',post(mm).GammaInv_i_m(:,k)'); % predictive densities (Gaussian)
                            else
                                predictives = predictives + post(mm).pi_i_m(k)*mvnpdf(gridSpace2D,post(mm).mu_i_m(:,k)',post(mm).GammaInv_i_m(:,:,k)); % predictive densities (Gaussian)
                            end
                        end
                %         predictives = predictives + (post(mm).lambda(k)/sumLambda)*pdfNonCentralT(gridSpace2D,post(mm).rho(:,k)',post(mm).Lambda(:,:,k),post(mm).degreesOfFreedom(k)); % predictive densities (t)
                        % or just use pi's instead of lambda's
                    end
                    gridSpace2DClusters = reshape(predictives,size(xGrid));

                    subplot(2,3,mm);
                    imagesc(xx,yy,gridSpace2DClusters);
                    % c = colormapRedblue;
                    % colormap(c);
                    % colorbar;
                    % caxis([0 .1])
                    set(gca,'Ydir','normal');
                    hold on;

                    plotTwoClusters2D(bags);
                    hold on;
                    for k=1:K
                        if (post(mm).pi_i_m(k)>self.pruneThreshold)
                            if self.featsUncorrelated
                                myPlotMvnEllipse(post(mm).mu_i_m(:,k)',diag(post(mm).GammaInv_i_m(:,k)),1,[],colorVector(mm));
                            else
                                myPlotMvnEllipse(post(mm).mu_i_m(:,k)',post(mm).GammaInv_i_m(:,:,k),1,[],colorVector(mm));
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
            
            subplot(2,3,4);
            Kmax = max([length(post(1).pi_i_m) length(post(2).pi_i_m)]);
            for mm=1:2
                stem(post(mm).pi_i_m,'fill','MarkerFaceColor',colorVector(mm),'Color',colorVector(mm),'LineWidth',2); grid on; grid minor;
                axis([1 Kmax+1 0 1]);
                hold on;
            end
            ylabel('\pi_i','FontWeight','b','FontSize',14);
            xlabel('components in a mixture model','FontWeight','b','FontSize',12);
            title(['nIteration=',num2str(nIteration)],'FontWeight','b','FontSize',12);
            legend('{\pi_i}^{1}','{\pi_i}^{0}','location','best');
            hold off;

            subplot(2,3,5);
            stem(post(3).eta_m,'fill','MarkerFaceColor',colorVector(mm),'Color','k','LineWidth',2); grid on;
            axis([1 2 0 1]);
            hold on;
            ylabel('\eta_{m}','FontWeight','b','FontSize',14);
            xlabel('mixture models','FontWeight','b','FontSize',12);
            hold off;

            % subplot(2,2,5); 
            % imagesc(eStep(1).phi_l_i_m); 
            % title(['\phi^{1}_{li}'],'FontWeight','b');%colorbar;
            % ylabel('instances','FontWeight','b');
            % xlabel('K^{1}','FontWeight','b');
            % 
            % subplot(2,2,6); 
            % imagesc(eStep(2).phi_l_i_m); 
            % title(['\phi^{0}_{li}'],'FontWeight','b');%colorbar;
            % ylabel('instances','FontWeight','b');
            % xlabel('K^{0}','FontWeight','b');
            % 
            % subplot(2,2,7); 
            % imagesc(eStep(3).phi_l_m_M); 
            % title(['\phi^{M}_{lm}'],'FontWeight','b');%colorbar;
            % ylabel('instances','FontWeight','b');
            % xlabel('m=\{1,0\}','FontWeight','b');
            % 
            subplot(2,3,6);
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

classdef prtClassMilVbDpMn < prtClass
    % prtClassMilVbDpMn

    properties (SetAccess=private)
        name = 'MilVbDpMn' % Fisher Linear Discriminant
        nameAbbreviation = 'MilVbDpMn'            % FLD
        isNativeMary = false;  % False
    end
    
    
    properties
        rndInit = 0;
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
		function self = prtClassMilVbDpMn(varargin)

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
            bags.NWords = sum(bags.data,2);
            
            % initialize parameters
            fprintf('Initializing parameters \n');
            [prior,eStep]  = vbMilMnInitialization(self,bags);
            vbIter = 1;
            converged  = 0;
            while converged==0   
                
                % M-step
                % Variational Posterior Approxiamtion of pi, mu, lamda (model parameters)
                post = vbMilMnUpdate(self,bags,prior,eStep);           
                                               
                self.rvH1 = post(1).rv;
                self.rvH0 = post(2).rv;
                self.mixOfRvs = post(3).mixOfRvs;
                
                % E-step
                eStep = vbMilMnExpectation(self,bags,eStep,post);
               
                % NFE
                nfeTerms(vbIter) = vbMilMnNegFreeEnergy(self,prior,post,eStep);
                if (vbIter>1)
                    nfeTerms(vbIter).nfe(:,2) = nfeTerms(vbIter).nfe(:,1)-nfeTerms(vbIter-1).nfe(:,1);
                    if (nfeTerms(vbIter).nfe(:,1)<nfeTerms(vbIter-1).nfe(:,1))
                        disp(['Warning - neg free energy decreased at nIteration=',num2str(vbIter)]);
                    end
                end
                
                % Plot learned clusters
                if (self.plot)
                    nfeTermsCat = concatStructs(1,nfeTerms);
                    handle = vbMilMnPlotLearnedClusters(self,bags,prior,post,eStep,nfeTermsCat.nfe(:,1),vbIter); % GMM model parameters (mu,Sigma) are Normal-Wishart distributed!
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
                
        function [prior,eStep] = vbMilMnInitialization(self,bags)
            
            [N,D] = size(bags.data);
            KH = [self.K1 self.K0];

            % Initialize Priors
            prior(3).alpha_m       = [.5,.5];  % eta_m~Dir(alpha_m) = mixing proportion of the H1/H0 mixture models 
            for mm=1:2 % mm = mixture model
                K = KH(mm);
                % Priors in Eqn 7
                prior(mm).gamma_0_1_m      = ones(1,K);          
                prior(mm).gamma_0_2_m      = 1e-3*ones(1,K); % sparsity inversely prop to shape parameter
                                                        % lower values = sparser model, Beta(1,1e-3) ~ 1, Beta(1,1) = U(0,1), Beta(1,1e+3) ~ 0                 
                vec1 = ones(D,1)/D;
                vec1Norm = vec1/sum(vec1);
                % Priors in Eqn 15
                prior(mm).omega_0_m     = repmat(vec1Norm,1,K); % Diffuse prior??
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
                eStep(1).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(2).phi_l_i_m,eStep(3).phi_l_m_M(:,2));

                % Update H0 Mix Model (M-step for only H0 mixture model)    
                post(2) = vbMilMnUpdateThisMixMod(self,bags,prior(2),eStep(2));         
                self.rvH0 = post(2).rv;

                % E-step only for H1 mixture model
                eStepThisMixMod = vbMilMnExpectationThisMixMod(self,bags,post(2),eStep(2),eStep(3).phi_l_m_M(:,2));
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

                % Initialize rH1H0 (big PHI)
                eStep(3).phi_l_m_M = zeros(N,2);
                eStep(3).phi_l_m_M(bags.myLabelH==1,1) = 1;
                eStep(3).phi_l_m_M(bags.myLabelH==0,2) = 1;

                % Weight phi_l_i_0 by phi_l_0_M
                eStep(1).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(1).phi_l_i_m,eStep(3).phi_l_m_M(:,1));

                % Update H1 Mix Model (M-step only for H1 mixture model)
                post(1) = vbMilMnUpdateThisMixMod(self,bags,prior(1),eStep(1));

                % E-step only for H1 mixture model
                eStepThisMixMod = vbMilMnExpectationThisMixMod(self,bags,post(1),eStep(1),eStep(3).phi_l_m_M(:,1));
                eStep(1).phi_l_i_m = eStepThisMixMod.phi_l_i_m;
            end    
            % Weight phi_l_i_0 by phi_l_0_M
            for mm=1:2
                eStep(mm).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(mm).phi_l_i_m,eStep(3).phi_l_m_M(:,mm));
            end 
   
        end
        
        function post = vbMilMnUpdate(self,bags,prior,eStep)  
            
            for mm=1:2
                % M-step for the mth mixture model
                post(mm) = vbMilMnUpdateThisMixMod(self,bags,prior(mm),eStep(mm));
            end               
            % Eqn 24
            post(3).N_m_bar   = sum(eStep(3).phi_l_m_M(bags.label==1,:),1);   
            post(3).N_m_bar(post(3).N_m_bar==0) = eps;
            post(3).alpha_m  = post(3).N_m_bar + prior(3).alpha_m;
            post(3).eta_m = post(3).alpha_m/sum(post(3).alpha_m,2);
            post(3).mixOfRvs = prtRvDiscrete('probabilities',post(3).eta_m);
            
        end

        function post = vbMilMnUpdateThisMixMod(self,bags,prior,eStep)    
            
            [~,D] = size(bags.data);
            K = length(prior.gamma_0_1_m);
            resp = eStep.phi_l_m_M_phi_l_i_m;   
            % Eqn 25
            post.N_i_m_bar = sum(resp,1);
            post.N_i_m_bar(post.N_i_m_bar==0) = eps;
            N_i_m_bar = post.N_i_m_bar;

            % mix
            sortObj = vbMilSortCount(N_i_m_bar); % If exchangeable, why sort? Why does sorting help?
            % Eqn 20
            post.gamma_i_1_m = prior.gamma_0_1_m + sortObj.sortedCounts;
            % Eqn 21
            post.gamma_i_2_m = prior.gamma_0_2_m + fliplr(cumsum(fliplr(sortObj.sortedCounts),2)) - sortObj.sortedCounts;
            post.pi_i_m = N_i_m_bar/sum(N_i_m_bar);

            % model
            for k=1:K      
                % Eqn 46
                post.omega_i_m(:,k) = prior.omega_0_m(:,k) + sum(bsxfun(@times,resp(:,k),bags.data),1)';
                probabilities = post.omega_i_m(:,k)/sum(post.omega_i_m(:,k)); % expectation of Dirichlet
                mixMod(k) = prtRvMultinomial('probabilities',probabilities');              
            end       
            post.rv = prtRvMixture('mixingProportions',post.pi_i_m,'components',mixMod);           
            
        end
        
        function eStep = vbMilMnExpectation(self,bags,eStep,post) 
            
            % Eqn 108
            eStep(3).expect_log_eta_m    = psi(0,post(3).alpha_m) - psi(0,sum(post(3).alpha_m));
            for mm=1:2
                eStepThisMixMod = vbMilMnExpectationThisMixMod(self,bags,post(mm),eStep(mm),eStep(3).phi_l_m_M(:,mm));
                eStep(mm).phi_l_i_m = eStepThisMixMod.phi_l_i_m;
                eStep(mm).phi_l_m_M_phi_l_i_m = eStepThisMixMod.phi_l_m_M_phi_l_i_m;
                eStep(mm).variationalAvgLL = eStepThisMixMod.variationalAvgLL;
                eStep(mm).expect_log_pi_i_m = eStepThisMixMod.expect_log_pi_i_m;
                % Eqn 58
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
                disp('\nNaNs in phi_l_m_M');
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

        function eStep = vbMilMnExpectationThisMixMod(self,bags,post,eStep,rH1H0ThisMixMod)         
            
            [N,D] = size(bags.data);
            logFactorialNWords = gammaln(bags.NWords+1);
            sumLogFactorialSamples = sum(gammaln(bags.data+1),2);

            K_m = length(post.pi_i_m);
            eStep.expect_log_pi_i_m        = nan(1,K_m);
            eStep.variationalAvgLL  = nan(N,K_m);
            eStep.log_phi_l_i_m_hat        = nan(N,K_m);

            % E-Step    
            % mix
            % Eqn 106
            eStep.expect_log_v_i_m_sorted            = psi(0,post.gamma_i_1_m) - psi(0,post.gamma_i_1_m+post.gamma_i_2_m);
            % Eqn 107
            eStep.expect_log_1_minus_v_i_m_sorted      = psi(0,post.gamma_i_2_m) - psi(0,post.gamma_i_1_m+post.gamma_i_2_m);
            % Eqn 105
            for k=1:K_m
                if (k==1)
                    eStep.expect_log_pi_i_m_sorted(k)        = eStep.expect_log_v_i_m_sorted(k);
                else
                    eStep.expect_log_pi_i_m_sorted(k)        = eStep.expect_log_v_i_m_sorted(k) + sum(eStep.expect_log_1_minus_v_i_m_sorted(1:k-1),2);
                end
            end    
            % Unsort
            sortObj = vbMilSortCount(post.N_i_m_bar);
            eStep.expect_log_pi_i_m = eStep.expect_log_pi_i_m_sorted(sortObj.unsortIndices);

            % Eqn 113
            eStep.expect_log_p_i_m = bsxfun(@minus,psi(0,post.omega_i_m),psi(0,sum(post.omega_i_m,1)));
            eStep.varAvgLLClusteriMixmPartTerm = bags.data*eStep.expect_log_p_i_m;
            % Eqn 120
            eStep.variationalAvgLL = bsxfun(@plus,logFactorialNWords-sumLogFactorialSamples,eStep.varAvgLLClusteriMixmPartTerm);
            
            if any(any((eStep.varAvgLLClusteriMixmPartTerm==-inf|eStep.varAvgLLClusteriMixmPartTerm==0)))
                keyboard;
                eStep.varAvgLLClusteriMixmPartTerm(eStep.varAvgLLClusteriMixmPartTerm==-inf|eStep.varAvgLLClusteriMixmPartTerm==0)=min(eStep.varAvgLLClusteriMixmPartTerm(eStep.varAvgLLClusteriMixmPartTerm~=0));
            end
            
            if any(any((eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)))
                keyboard;
                eStep.variationalAvgLL(eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)=min(eStep.variationalAvgLL(eStep.variationalAvgLL~=0));
            end

            % Eqn 53
            eStep.log_phi_l_i_m_hat = bsxfun( @plus,eStep.expect_log_pi_i_m,bsxfun(@times,rH1H0ThisMixMod,eStep.varAvgLLClusteriMixmPartTerm) );  % sim to eq (13) % [N*1]

            if ( any(isinf(eStep.log_phi_l_i_m_hat(:,k))) || any(isnan(eStep.log_phi_l_i_m_hat(:,k))) ) % check for inf/nan
                disp('Error - inf/nan in logResp values');
                keyboard;
            end

            % Normalize % Eqn 54
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
        
        function nfeTerms = vbMilMnNegFreeEnergy(self,prior,post,eStep)
             
            KLz = zeros(1,2);
            weightedVariationalAvgLL = zeros(1,2);
            for mm=1:2
                weightedVariationalAvgLL(mm) = sum(sum(eStep(mm).phi_l_m_M_phi_l_i_m .* eStep(mm).variationalAvgLL,1),2);

                respSmallzInfoTerm = eStep(mm).phi_l_m_M_phi_l_i_m.*log(eStep(mm).phi_l_i_m);
                respSmallzInfoTerm(eStep(mm).phi_l_m_M_phi_l_i_m==0) = 0;

                expectLogPiTerm = post(mm).N_i_m_bar.*eStep(mm).expect_log_pi_i_m;
                % Eqn 123
                KLz(mm) = sum( (sum(respSmallzInfoTerm,1) - expectLogPiTerm), 2);
%                 KLz(mm) = sum( sum(respSmallzInfoTerm,1) - suffStat(mm).N_i_m_bar.*eStep(mm).expect_log_pi_i_m, 2);
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
            mthKLP = zeros(1,2);
            for mm=1:2 % mm = nth mixture model
                K = length(post(mm).pi_i_m);
                KLv       = zeros(1,K);
                KLP       = zeros(1,K);
                for k=1:K
                    KLv(k) = KLBeta(post(mm).gamma_i_1_m(k),post(mm).gamma_i_2_m(k),prior(mm).gamma_0_1_m(k),prior(mm).gamma_0_2_m(k));
                    KLP(k) = KLDirichlet(post(mm).omega_i_m(:,k),prior(mm).omega_0_m(:,k));
                end
                mthKLv(mm) = sum(KLv);
                mthKLP(mm) = sum(KLP);
            end

            sumKLv = sum(mthKLv);
            sumKLP = sum(mthKLP);
            
            F = sumWeightedVariationalAvgLL + sumKLZeta - sumKLeta + sumKLz - sumKLv - sumKLP;

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
            nfeTerms.sumKLP     = sumKLP;            
            
        end                
        
        function fig1 = vbMilMnPlotLearnedClusters(self,bags,prior,post,eStep,negFreeEnergy,nIteration)
            
%             [~,post] = vbDpMnMilPrune(prior,post,myOptions);

            fig1 = figure(1);

%             subplot(3,3,1)
%             imagesc(bags.data');
%             ylabel('features','FontWeight','b');
%             xlabel('samples','FontWeight','b');
% 
%             MM = [1 0];
%             for mm=1:2
%                 subplot(3,3,1+mm)
%                 imagesc(post(mm).omega_i_m);
%                 ylabel('features','FontWeight','b');
%                 xlabel(['\Omega^{',num2str(MM(mm)),'}_{k}'],'FontWeight','b');
%             end

            colorVector = ['r','b'];
            subplot(1,3,1);
            for mm=1:2
                K(mm) = length(post(mm).pi_i_m);
                stem(post(mm).pi_i_m,'fill','MarkerFaceColor',colorVector(mm),'Color',colorVector(mm),'LineWidth',2); grid on; grid minor;
                hold on;
            end
            % keyboard;
            axis([1 max(K) 0 1]);
            ylabel('\pi_k','FontWeight','b');
            xlabel('mixture component','FontWeight','b');
            legend('{\pi_k}^H1','{\pi_k}^H0','location','best');
            hold off;

            subplot(1,3,2);
            stem(post(3).eta_m,'fill','MarkerFaceColor',colorVector(mm),'Color','k','LineWidth',2); grid on;
            axis([1 2 0 1]);
            hold on;
            ylabel('\eta_H','FontWeight','b');
            xlabel('mixture model component','FontWeight','b');
            hold off;  

            subplot(1,3,3);
            plot(negFreeEnergy,'.'); grid on;
            ylabel('Neg Free Energy','FontWeight','b');
            xlabel('nIteration','FontWeight','b');

%             MM = [1,0];
%             for mm=1:2
%                 subplot(3,3,6+mm)
%                 area(eStep(mm).phi_l_i_m,'edgecolor','none')
%                 ylim([0 1]);
%                 xlim([1 size(eStep(mm).phi_l_i_m,1)]);
%                 ylabel(['rH',num2str(MM(mm))],'FontWeight','b');
%                 xlabel('instances','FontWeight','b');
%             end
% 
%             subplot(3,3,9)
%             area(eStep(3).phi_l_m_M,'edgecolor','none')
%             ylim([0 1]);
%             xlim([1 size(eStep(3).phi_l_m_M,1)]);
%             ylabel('rH1H0','FontWeight','b');
%             xlabel('instances','FontWeight','b');
            
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

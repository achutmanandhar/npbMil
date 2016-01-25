classdef prtClassMilVbDpMnOldNotations < prtClass
    % prtClassMilVbDpMnOldNotations

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
		function self = prtClassMilVbDpMnOldNotations(varargin)

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
            prior(3).alpha       = [.5,.5];  % eta~Dir(alpha) = mixing proportion of the H1/H0 mixture models 
            for mm=1:2 % mm = mixture model
                K = KH(mm);
                prior(mm).gamma1      = ones(1,K);          % v|pi~Beta(gamma1,gamma2), shape parameters
                prior(mm).gamma2      = 1e-3*ones(1,K);       % sparsity inversely prop to shape parameter
                                                      % lower values = sparser model, Beta(1,1e-3) ~ 1, Beta(1,1) = U(0,1), Beta(1,1e+3) ~ 0 
                % Mn priors
                vec1 = ones(D,1)/D;
                vec1Norm = vec1/sum(vec1);
                prior(mm).omega     = repmat(vec1Norm,1,K); % Diffuse prior??
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
                post(2) = vbMilMnUpdateThisMixMod(self,bags,prior(2),eStep(2));         
                self.rvH0 = post(2).rv;

                % Compute p(x|H0 MM) for all instances
                eStepThisMixMod = vbMilMnExpectationThisMixMod(self,bags,post(2),eStep(2),eStep(3).rH1H0(:,2));
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
                post(1) = vbMilMnUpdateThisMixMod(self,bags,prior(1),eStep(1));

                % Compute p(x|H0 MM) for all instances
                eStepThisMixMod = vbMilMnExpectationThisMixMod(self,bags,post(1),eStep(1),eStep(3).rH1H0(:,1));
                eStep(1).rH = eStepThisMixMod.rH;
            end    
            % weight small phi's by big PHI
            for mm=1:2
                eStep(mm).rHrH1H0 = bsxfun(@times,eStep(mm).rH,eStep(3).rH1H0(:,mm));
            end 
   
        end
        
        function post = vbMilMnUpdate(self,bags,prior,eStep)  
            
            for mm=1:2
                post(mm) = vbMilMnUpdateThisMixMod(self,bags,prior(mm),eStep(mm));
            end                   
            post(3).countsH   = sum(eStep(3).rH1H0(bags.label==1,:),1);   
            post(3).countsH(post(3).countsH==0) = eps;
            post(3).alpha  = post(3).countsH + prior(3).alpha;
            post(3).eta = post(3).alpha/sum(post(3).alpha,2);
            post(3).mixOfRvs = prtRvDiscrete('probabilities',post(3).eta);
            
        end

        function post = vbMilMnUpdateThisMixMod(self,bags,prior,eStep)    
            
            [~,D] = size(bags.data);
            K = length(prior.gamma1);
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
            for k=1:K             
                post.omega(:,k) = prior.omega(:,k) + sum(bsxfun(@times,resp(:,k),bags.data),1)';
                probabilities = post.omega(:,k)/sum(post.omega(:,k)); % expectation of Dirichlet
                mixMod(k) = prtRvMultinomial('probabilities',probabilities');              
            end       
            post.rv = prtRvMixture('mixingProportions',post.pi,'components',mixMod);           
            
        end
        
        function eStep = vbMilMnExpectation(self,bags,eStep,post) 
            
            eStep(3).logEtaTilda    = psi(0,post(3).alpha) - psi(0,sum(post(3).alpha));
            for mm=1:2
                eStepThisMixMod = vbMilMnExpectationThisMixMod(self,bags,post(mm),eStep(mm),eStep(3).rH1H0(:,mm));
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

        function eStep = vbMilMnExpectationThisMixMod(self,bags,post,eStep,rH1H0ThisMixMod)         
            
            [N,D] = size(bags.data);
            logFactorialNWords = gammaln(bags.NWords+1);
            sumLogFactorialSamples = sum(gammaln(bags.data+1),2);

            K = length(post.pi);
            eStep.logPiTilda        = nan(1,K);
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
            sortObj = vbMilSortCount(post.counts);
            eStep.logPiTilda = eStep.logPiTildaSorted(sortObj.unsortIndices);

            eStep.logPTilda = bsxfun(@minus,psi(0,post.omega),psi(0,sum(post.omega,1)));
            eStep.varAvgLLClusteriMixmPartTerm = bags.data*eStep.logPTilda;
            eStep.variationalAvgLL = bsxfun(@plus,logFactorialNWords-sumLogFactorialSamples,eStep.varAvgLLClusteriMixmPartTerm);
            
            if any(any((eStep.varAvgLLClusteriMixmPartTerm==-inf|eStep.varAvgLLClusteriMixmPartTerm==0)))
                keyboard;
                eStep.varAvgLLClusteriMixmPartTerm(eStep.varAvgLLClusteriMixmPartTerm==-inf|eStep.varAvgLLClusteriMixmPartTerm==0)=min(eStep.varAvgLLClusteriMixmPartTerm(eStep.varAvgLLClusteriMixmPartTerm~=0));
            end
            
            if any(any((eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)))
                keyboard;
                eStep.variationalAvgLL(eStep.variationalAvgLL==-inf|eStep.variationalAvgLL==0)=min(eStep.variationalAvgLL(eStep.variationalAvgLL~=0));
            end

            eStep.logrHTilda = bsxfun( @plus,eStep.logPiTilda,bsxfun(@times,rH1H0ThisMixMod,eStep.varAvgLLClusteriMixmPartTerm) );  % sim to eq (13) % [N*1]

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
        
        function nfeTerms = vbMilMnNegFreeEnergy(self,prior,post,eStep)
             
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
            mthKLP = zeros(1,2);
            for mm=1:2 % mm = nth mixture model
                K = length(post(mm).pi);
                KLv       = zeros(1,K);
                KLP       = zeros(1,K);
                for k=1:K
                    KLv(k) = KLBeta(post(mm).gamma1(k),post(mm).gamma2(k),prior(mm).gamma1(k),prior(mm).gamma2(k));
                    KLP(k) = KLDirichlet(post(mm).omega(:,k),prior(mm).omega(:,k));
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

            subplot(3,3,1)
            imagesc(bags.data');
            ylabel('features','FontWeight','b');
            xlabel('samples','FontWeight','b');

            MM = [1 0];
            for mm=1:2
                subplot(3,3,1+mm)
                imagesc(post(mm).omega);
                ylabel('features','FontWeight','b');
                xlabel(['\Omega^{',num2str(MM(mm)),'}_{k}'],'FontWeight','b');
            end

            colorVector = ['r','b'];
            subplot(3,3,4);
            for mm=1:2
                K(mm) = length(post(mm).pi);
                stem(post(mm).pi,'fill','MarkerFaceColor',colorVector(mm),'Color',colorVector(mm),'LineWidth',2); grid on; grid minor;
                hold on;
            end
            % keyboard;
            axis([1 max(K) 0 1]);
            ylabel('\pi_k','FontWeight','b');
            xlabel('mixture component','FontWeight','b');
            legend('{\pi_k}^H1','{\pi_k}^H0','location','best');
            hold off;

            subplot(3,3,5);
            stem(post(3).eta,'fill','MarkerFaceColor',colorVector(mm),'Color','k','LineWidth',2); grid on;
            axis([1 2 0 1]);
            hold on;
            ylabel('\eta_H','FontWeight','b');
            xlabel('mixture model component','FontWeight','b');
            hold off;  

            subplot(3,3,6);
            plot(negFreeEnergy,'.'); grid on;
            ylabel('Neg Free Energy','FontWeight','b');
            xlabel('nIteration','FontWeight','b');

%             MM = [1,0];
%             for mm=1:2
%                 subplot(3,3,6+mm)
%                 area(eStep(mm).rH,'edgecolor','none')
%                 ylim([0 1]);
%                 xlim([1 size(eStep(mm).rH,1)]);
%                 ylabel(['rH',num2str(MM(mm))],'FontWeight','b');
%                 xlabel('instances','FontWeight','b');
%             end
% 
%             subplot(3,3,9)
%             area(eStep(3).rH1H0,'edgecolor','none')
%             ylim([0 1]);
%             xlim([1 size(eStep(3).rH1H0,1)]);
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

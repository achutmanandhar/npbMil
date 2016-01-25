classdef prtClassMilVbDpGmmPeteCommentsAchut < prtClass
    % prtClassMilVbDpGmm

    properties (SetAccess=private)
        name = 'MilVbDpGmm' % Fisher Linear Discriminant
        nameAbbreviation = 'MilVbDpGmm'            % FLD
        isNativeMary = false;  % False
    end
    
    
    properties
        K0
        K1
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
            
            milStruct = dsMil.data;
            x = cat(1,milStruct.data);
            
            bagLabels = dsMil.targets;
            expandedBagLabels = dsMil.expandedTargets;
            
            bagIndices = dsMil.getBagInds;
            uniqueBagIndices = unique(bagIndices);
            
            d = size(x,2);
            
            params = struct('K0',5,'K1',5,'gamma_0_1_m0',1,'gamma_0_2_m0',.001,'gamma_0_1_m1',1,'gamma_0_2_m1',.001,...
                'v_0_m',d+2,'v_1_m',d+2,'beta_1_m',d,'beta_0_m',d, ...
                'rho_0_1',zeros(1,d),'rho_0_0',zeros(1,d),'Phi_0_1',eye(d)*d,'Phi_0_0',eye(d)*d,'alpha',[.5 .5]); 
            % Strong prior 'alpha',[9 1]*1000 ;           
            
            [rvH0,rvH1,xH0,xH1] = initialize(self,{milStruct.data},bagLabels,params);
            
            [p0,pp] = rvH0.pdf(x);
            phi_0_i = bsxfun(@rdivide,pp,sum(pp,2));
            
            [p1,pp] = rvH1.pdf(x);
            phi_1_i = bsxfun(@rdivide,pp,sum(pp,2));
            phi_1_i(expandedBagLabels == 0,:) = 0;
            p1(expandedBagLabels ==0) = 0;
            
            phi_1_M = p1./(p0 + p1);
            phi_0_M = 1-phi_1_M;           
            
%             % 12/08/14 Modified by AM
%             load('C:\Users\manandhar\research\matlabStuff\Research\npbMil\PhiInitExp\msrcorid\PhiVb01\eStepAfterInitializationIncest');                        
%             phi_0_i = eStep(2).rH;
%             phi_1_i = eStep(1).rH;
%             phi_1_M = eStep(3).rH1H0(:,1);
%             phi_0_M = eStep(3).rH1H0(:,2);
            
            locEps = 1e-6;
            
            for vbIter = 1:500
                %Equations 67 and 68; note, these are different from 20-21 (!)
                %     gamma_i_1_m0 = sum(bsxfun(@times,phi_0_M,phi_0_i)) + params.gamma_0_1_m0;
                %     gamma_i_2_m0 = cat(2,0,cumsum(gamma_i_1_m0(1:end-1),2)) + params.gamma_0_2_m0;
                %
                %     gamma_i_1_m1 = sum(bsxfun(@times,phi_1_M,phi_1_i)) + params.gamma_0_1_m1;
                %     gamma_i_2_m1 = cat(2,0,cumsum(gamma_i_1_m1(1:end-1),2)) + params.gamma_0_2_m1;             
                
                % I sort the counts, sum(bsxfun(@times,phi_0_M,phi_0_i)),
                % before using them to update the gamma's, gamma_i_1_m0 and
                % gamma_i_2_m0
                % Yes, it should be mentioned in the paper as well.
                
                %Equations 20-21
                gamma_i_1_m0 = sum(bsxfun(@times,phi_0_M,phi_0_i)) + params.gamma_0_1_m0;
                gamma_i_2_m0 = fliplr(cumsum(fliplr(gamma_i_1_m0))) - gamma_i_1_m0 + params.gamma_0_2_m0;
                
                gamma_i_1_m1 = sum(bsxfun(@times,phi_1_M,phi_1_i)) + params.gamma_0_1_m1;
                gamma_i_2_m1 = fliplr(cumsum(fliplr(gamma_i_1_m1))) - gamma_i_1_m1 + params.gamma_0_2_m1;

                %Update priors... H0 Gaussian; Equations 77-79 (H0)
                respMat = bsxfun(@times,phi_0_M,phi_0_i);
                nBar0 = sum(respMat)+locEps;  %avoid 1./0 problems
              
                for cluster = 1:params.K0
                    mu_i_0(cluster,:) = 1/nBar0(cluster)*sum(bsxfun(@times,respMat(:,cluster),x));
                    
                    xx = bsxfun(@times,sqrt(respMat(:,cluster)),bsxfun(@minus,x,mu_i_0(cluster,:)));
                    c = 1/nBar0(cluster)*xx'*xx;
                    cov_i_0(cluster,:) = c(:);
                end
                
                %Update priors... H1 Gaussian; Equations 77-79
                respMat = bsxfun(@times,phi_1_M,phi_1_i);
                nBar1 = sum(respMat)+locEps;
                for cluster = 1:params.K1
                    mu_i_1(cluster,:) = 1/nBar1(cluster)*sum(bsxfun(@times,respMat(:,cluster),x));
                    
                    xx = bsxfun(@times,sqrt(respMat(:,cluster)),bsxfun(@minus,x,mu_i_1(cluster,:)));
                    c = 1/nBar1(cluster)*xx'*xx;
                    cov_i_1(cluster,:) = c(:);
                end
                
                %Updating hyper-parameters; eqns 81-84
                nu_0 = params.v_0_m + nBar0;
                nu_1 = params.v_1_m + nBar1;
                
                beta_0 = params.beta_0_m + nBar0;
                beta_1 = params.beta_1_m + nBar1;
                
                
                for cluster = 1:params.K0
                    rho_0(cluster,:) = params.beta_0_m*params.rho_0_0 + nBar0(cluster)*mu_i_0(cluster,:);
                    rho_0(cluster,:) = rho_0(cluster,:)./beta_0(cluster);
                    
                    %See equation 84; Note: this is different from Equation 84 - 84 is
                    %wrong
                    %covPart = nBar0(cluster) * reshape(cov_i_0(cluster,:),d,d) * ((mu_i_0(cluster,:) - rho_0(cluster,:))'*(mu_i_0(cluster,:) - rho_0(cluster,:)));
                    covPart = nBar0(cluster) * params.beta_0_m * ((mu_i_0(cluster,:) - params.rho_0_0)'*(mu_i_0(cluster,:) - params.rho_0_0));
                    covPart = covPart ./ beta_0(cluster);
                    
                    phiCov = params.Phi_0_0 + nBar0(cluster)*reshape(cov_i_0(cluster,:),d,d) + covPart;
                    Phi_0(cluster,:) = phiCov(:)';
                end
                
                for cluster = 1:params.K1
                    rho_1(cluster,:) = params.beta_1_m*params.rho_0_1 + nBar1(cluster)*mu_i_1(cluster,:);
                    rho_1(cluster,:) = rho_1(cluster,:)./beta_1(cluster);
                    
                    %See equation 84; Note: this is different from Equation 84 - 84 is
                    %wrong
                    %covPart = nBar1(cluster) * reshape(cov_i_1(cluster,:),d,d) * ((mu_i_1(cluster,:) - rho_1(cluster,:))'*(mu_i_1(cluster,:) - rho_1(cluster,:)));
                    %         covPart = nBar1(cluster) * reshape(cov_i_1(cluster,:),d,d) * ((mu_i_1(cluster,:) - params.rho_0_0)'*(mu_i_1(cluster,:) - params.rho_0_0));
                    covPart = nBar1(cluster) * params.beta_1_m * ((mu_i_1(cluster,:) - params.rho_0_1)'*(mu_i_1(cluster,:) - params.rho_0_1));
                    covPart = covPart ./ beta_1(cluster);
                    
                    phiCov = params.Phi_0_1 + nBar1(cluster)*reshape(cov_i_1(cluster,:),d,d) + covPart;
                    Phi_1(cluster,:) = phiCov(:)';
                end
                
                % <log_pi>  Equation 102
                matGamma = digamma(gamma_i_1_m0) - digamma(gamma_i_1_m0 + gamma_i_2_m0);
                matGamma2 = digamma(gamma_i_2_m0) - digamma(gamma_i_1_m0 + gamma_i_2_m0);
                log_pi_0 = matGamma + cat(2,0,cumsum(matGamma2(1:end-1)));
                
                matGamma = digamma(gamma_i_1_m1) - digamma(gamma_i_1_m1 + gamma_i_2_m1);
                matGamma2 = digamma(gamma_i_2_m1) - digamma(gamma_i_1_m1 + gamma_i_2_m1);
                log_pi_1 = matGamma + cat(2,0,cumsum(matGamma2(1:end-1)));   
                
                % <Log \eta_m>  Equation 105
                nBarM0 = params.alpha(1) +  sum(phi_0_M(expandedBagLabels == 1,:));
                nBarM1 = params.alpha(2) +  sum(phi_1_M(expandedBagLabels == 1,:));
                log_eta = digamma([nBarM0,nBarM1]) - digamma(nBarM1+nBarM0);
                
                % 12/08/13 Modified by AM, 
                % An internal VB iteration to get a better estimate of 
                % phi_1_M, phi_0_M, phi_1_i, phi_0_i since they are dependent
                
%                 for nVbInternalIter=1:1
                    
                    % <log det ||>
                    for cluster = 1:params.K0
                        temp_det0 = sum(digamma((nu_0(cluster) + 1 - (1:d))/2));
                        temp_det0 = temp_det0 + d*log(2);
                        log_det_0(cluster) = temp_det0 + log(det(reshape(Phi_0(cluster,:),d,d)^-1));
                    end
                    for cluster = 1:params.K1
                        temp_det1 = sum(digamma((nu_1(cluster) + 1 - (1:d))/2));
                        temp_det1 = temp_det1 + d*log(2);
                        log_det_1(cluster) = temp_det1 + log(det(reshape(Phi_1(cluster,:),d,d)^-1));
                    end                

                    %Equatioin 113
                    for cluster = 1:params.K0
                        c = reshape(Phi_0(cluster,:),d,d);
                        %         v = prtUtilCalcDiagXcInvXT(bsxfun(@minus,x,mu_i_0(cluster,:)),c);
                        v = prtUtilCalcDiagXcInvXT(bsxfun(@minus,x,rho_0(cluster,:)),c);
                        inner_0_n(:,cluster) = d/beta_0(cluster) + nu_0(cluster).*v;
                    end

                    for cluster = 1:params.K1
                        c = reshape(Phi_1(cluster,:),d,d);
                        %         v = prtUtilCalcDiagXcInvXT(bsxfun(@minus,x,mu_i_1(cluster,:)),c);
                        v = prtUtilCalcDiagXcInvXT(bsxfun(@minus,x,rho_1(cluster,:)),c);
                        inner_1_n(:,cluster) = d/beta_1(cluster) + nu_1(cluster).*v;
                    end

                    for cluster = 1:params.K0
                        eLogPdf0(:,cluster) = -d*log(2*pi) + log_det_0(cluster) - inner_0_n(:,cluster);
                    end
                    eLogPdf0 = eLogPdf0*1/2;

                    for cluster = 1:params.K1
                        eLogPdf1(:,cluster) = -d*log(2*pi) + log_det_1(cluster) - inner_1_n(:,cluster);
                    end
                    eLogPdf1 = eLogPdf1*1/2;

                    % 12/08/13 Modified by AM, Eqn 50
%                     phiHat0 = bsxfun(@plus,eLogPdf0,log_pi_0);
%                     phiHat1 = bsxfun(@plus,eLogPdf1,log_pi_1);
                    phiHat0 = bsxfun(@plus,bsxfun(@times,phi_0_M,eLogPdf0),log_pi_0);
                    phiHat1 = bsxfun(@plus,bsxfun(@times,phi_1_M,eLogPdf1),log_pi_1);

                    phiHat1(expandedBagLabels == 0,:) = -inf;

                    phi_1_i = bsxfun(@rdivide,exp(phiHat1),sum(exp(phiHat1),2));
                    phi_0_i = bsxfun(@rdivide,exp(phiHat0),sum(exp(phiHat0),2));

                    phi_1_i(exp(phiHat1) == 0) = 0;
                    phi_0_i(exp(phiHat0) == 0) = 0;

                    % 12/08/13 Modified by AM, Eqn 55
%                     n1 = (log(sum(exp(phiHat1),2))+log_eta(2));
%                     n0 = (log(sum(exp(phiHat0),2))+log_eta(1));
                    n1 = sum(bsxfun(@times,phi_1_i,eLogPdf1),2) + log_eta(2);
                    n0 = sum(bsxfun(@times,phi_0_i,eLogPdf0),2) + log_eta(1);
                    
                    phi_1_M = exp(n1)./(exp(n1)+exp(n0));
                    phi_1_M(expandedBagLabels == 0) = 0;

                    phi_0_M = 1-phi_1_M;
                
%                 end % end of internal VB iteration
                
                for i = 1:size(rho_0,1)
                    c = Phi_0(i,:)./nu_0(i);
                    c = reshape(c,d,d);
                    plotMvnEllipse(rho_0(i,1:2),c(1:2,1:2),1);
                    mm0(i) = prtRvMvn('mu',rho_0(i,:),'sigma',c);
                end
                for i = 1:size(rho_1,1)
                    c = Phi_1(i,:)./nu_1(i);
                    c = reshape(c,d,d);
                    plotMvnEllipse(rho_1(i,1:2),c(1:2,1:2),1);
                    mm1(i) = prtRvMvn('mu',rho_1(i,:),'sigma',c);
                end
                
                pi0 = exp(log_pi_0); pi0 = pi0./sum(pi0);
                rv0 = prtRvGmm('nComponents',size(mu_i_0,1),'mixingProportions',pi0,'components',mm0);
                
                pi1 = exp(log_pi_1); pi1 = pi1./sum(pi1);
                rv1 = prtRvGmm('nComponents',size(mu_i_1,1),'mixingProportions',pi1,'components',mm1);
                
                
                self.rvH1 = rv1;
                self.rvH0 = rv0;
                
                
                if ~mod(vbIter,10);
                    disp(vbIter)
                    subplot(2,2,1);
                    stem(exp(log_pi_0));
                    subplot(2,2,2);
                    stem(exp(log_pi_1));
                    fprintf('exp(log_pi_0): %.2f\n',exp(log_pi_0))
                    fprintf('exp(log_pi_1): %.2f\n',exp(log_pi_1))
                    
                    subplot(2,2,3:4);
                    plot(x(expandedBagLabels == 0,1),x(expandedBagLabels == 0,2),'b.');
                    hold on;
                    plot(x(expandedBagLabels == 1,1),x(expandedBagLabels == 1,2),'r.');
                    
                    for i = 1:size(rho_0,1)
                        c = Phi_0(i,:)./nu_0(i);
                        c = reshape(c,d,d);
                        plotMvnEllipse(rho_0(i,1:2),c(1:2,1:2),1);
                        mm0(i) = prtRvMvn('mu',rho_0(i,:),'sigma',c);
                    end
                    for i = 1:size(rho_1,1)
                        c = Phi_1(i,:)./nu_1(i);
                        c = reshape(c,d,d);
                        plotMvnEllipse(rho_1(i,1:2),c(1:2,1:2),1);
                        mm1(i) = prtRvMvn('mu',rho_1(i,:),'sigma',c);
                    end
                    hold on; h = plot(rho_0(:,1),rho_0(:,2),'ko'); set(h,'MarkerFaceColor','k');
                    hold on; h = plot(rho_1(:,1),rho_1(:,2),'go'); set(h,'MarkerFaceColor','g');
                    
                    hold off;
                    
                    drawnow;
                end
                
            end
exp(log_eta)            
% keyboard;
            
        end
        
        function yOut = runAction(self,dsMil)
            
            for n = 1:dsMil.nObservations            
                milStruct = dsMil.data(n);
                data = milStruct.data; 
                
                h1 = self.rvH1.logPdf(data);
                h0 = self.rvH0.logPdf(data);
                
                % 12/08/14 Modified by AM
%                 l0 = h0'-prtUtilSumExp([h0';h1']);
%                 
%                 h0LogProbabilityBag = sum(l0);
%                 %exp(h0LogProbabilityBag)
%                 
%                 y(n,1) = 1-exp(h0LogProbabilityBag);

                % self object should have a component that describes the
                % mixing of the two prtRvGmm objects (Please refer to
                % Kenny's code prtBrvMultipleInstanceMixtures)
                
                % Eqn 46
                h1Weighted = h1 + log(.1); % log_eta(2) = log(.1)
                h0Weighted = h0 + log(.9); % log_eta(1) = log(.9)
                
                instancesInThisBagLLGivenPosBag = prtUtilSumExp(cat(2,h1Weighted,h0Weighted)')';
                thisBagLLGivenPosBag = sum(instancesInThisBagLLGivenPosBag,1);
                thisBagLLGivenNegBag = sum(h0,1);
                
                bagLL = [thisBagLLGivenPosBag,thisBagLLGivenNegBag] - prtUtilSumExp([thisBagLLGivenPosBag,thisBagLLGivenNegBag]')';
                y(n,1) = bagLL(:,1)-bagLL(:,2);

            end
            
            yOut = prtDataSetClass(y,dsMil.targets);
        end
        
        function [h0Mix,h1Mix,xH0,xH1] = initialize(self,xBag,bagLabels,params)
            
            %H0 Kmeans
            xH0 = cat(1,xBag{bagLabels == 0});
            idx0 = kmeans(xH0,params.K0,'Replicates',50,'EmptyAction','singleton','MaxIter',100);
            uIds = unique(idx0);
            
            for idx = 1:length(uIds)
                rvH0struct(idx).mean = mean(xH0(idx0 == idx,:),1);
                
                try
                    rvH0struct(idx).cov = cov(xH0(idx0 == idx,:));
                    chol(rvH0struct(idx).cov);
                    assert(size(xH0,2) == size(rvH0struct(idx).cov,2)); 
                catch ME
                    rvH0struct(idx).cov = diag(var(xH0(idx0 == idx,:))) + eye(size(xH0,2));
                end
                
                rvH0(idx) = prtRvMvn('mu',rvH0struct(idx).mean,'sigma',rvH0struct(idx).cov);
                pi(idx) = sum(idx0 == idx)./length(idx0);
                
            end
            pi = pi./sum(pi);
            h0Mix = prtRvGmm('nComponents',length(rvH0),'mixingProportions',pi,'components',rvH0);
            
            %Least likely H1
            h1Bags = xBag(bagLabels == 1);
            for h1Ind = 1:length(h1Bags)
                ll = h0Mix.logPdf(h1Bags{h1Ind});
                [~,ind] = min(ll);
                xH1(h1Ind,:) = h1Bags{h1Ind}(ind,:);
            end
            
            %H1 Kmeans
            idx1 = kmeans(xH1,params.K1,'Replicates',50,'EmptyAction','singleton','MaxIter',100);
            uIds = unique(idx1);
            
            pi = [];
            for idx = 1:length(uIds)
                rvH1struct(idx).mean = mean(xH1(idx1 == idx,:),1);
                
                try
                    rvH1struct(idx).cov = cov(xH1(idx0 == idx,:));
                    chol(rvH1struct(idx).cov);
                catch ME
                    rvH1struct(idx).cov = diag(var(xH1)) + eye(size(xH1,2));
                end
                
                rvH1(idx) = prtRvMvn('mu',rvH1struct(idx).mean,'sigma',rvH1struct(idx).cov);
                pi(idx) = sum(idx1 == idx)./length(idx1);
                
            end
            pi = pi./sum(pi);
            h1Mix = prtRvGmm('nComponents',length(rvH1),'mixingProportions',pi,'components',rvH1);
        end
    end
end

function [post nfeTerms] = npbMilMn(varargin)
% post = npbMil(varargin)

[bags KH1 KH0 myOptions] = parseInputs(varargin);

% for bag labels consistency
bags.label(bags.label==-1,:) = 0;

% initialize parameters
fprintf('Initializing parameters \n');
[prior,eStep]  = npbMilInitialization(bags,KH1,KH0,myOptions);

fprintf('Iterating VB Updates \n');
nIteration = 1; 
% frameCount = 1;
converged  = 0;
while (converged==0)
    
    % M-step
    % Variational Posterior Approxiamtion of pi, mu, lamda (model parameters)
    post = npbMilMnUpdate(bags,prior,eStep,myOptions);

    % Prune ineffective components
    if (myOptions.pruneClusters)
        [prior,post] = npbMilPrune(prior,post,myOptions);
    end

    % E-step
    eStep = npbMilMnExpectation(bags,eStep,post,myOptions); % Need eStep for eStep??? 
    
    % neg free energy
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

    sumKLZeta = -sum(KLZeta);
    sumKLz = -sum(KLz);
    sumVariationalAvgLL = sum(variationalAvgLL);

    nfeTerms(nIteration)   = npbMilNegFreeEnergy(prior,post,Lav,sumKLZeta,sumKLz,sumVariationalAvgLL,myOptions);
    if (nIteration>1)
        nfeTerms(nIteration).nfe(:,2) = nfeTerms(nIteration).nfe(:,1)-nfeTerms(nIteration-1).nfe(:,1);
        if (nfeTerms(nIteration).nfe(:,1)<nfeTerms(nIteration-1).nfe(:,1))
            disp(['Warning - neg free energy decreased at nIteration=',num2str(nIteration)]);
        end
    end

    if nIteration>10
        nfeTermsCat = concatStructs(1,nfeTerms);
        [converged, differentialPercentage, signedPercentage] = isconverged(nfeTermsCat.nfe(nIteration),nfeTermsCat.nfe(nIteration-1),myOptions.convergenceThreshold);
    end

    % Plot learned clusters
    if (myOptions.plot)
        nfeTermsCat = concatStructs(1,nfeTerms);
        handle = npbMilPlotLearnedClusters(bags,prior,post,eStep,nfeTermsCat.nfe(:,1),nIteration,myOptions); % GMM model parameters (mu,Sigma) are Normal-Wishart distributed!
    end
  
%     if (nIteration>1)
%         frameMovie(frameCount) = getframe(handle);
%         frameCount=frameCount+1;
%     end

%     nfeTermsCat = concatStructs(1,nfeTerms);
%     figure(1);
%     plot(nfeTermsCat.nfe,'.');grid on;

    if (nIteration>1)
        if (converged==1)
            fprintf('Convergence reached. Change in negative free energy below %.2e. \n',myOptions.convergenceThreshold);
        else
            nfeTermsCat = concatStructs(1,nfeTerms);
            fprintf('Negative Free Energy: %.2f, \t %.2e Change \n',nfeTermsCat.nfe(nIteration,1),nfeTermsCat.nfe(nIteration,2));
        end
    end
    
    nIteration = nIteration+1;
    if (nIteration==myOptions.maxIteration+1)
        converged=1;
    end
end 

% Prune ineffective components
if (myOptions.pruneClusters)
    [~,post] = npbMilPrune(prior,post,myOptions);
end

% % Plot learned clusters
% if (myOptions.plot)
    nfeTermsCat = concatStructs(1,nfeTerms);
    handle = npbMilPlotLearnedClusters(bags,prior,post,eStep,nfeTermsCat.nfe(:,1),nIteration,myOptions); % GMM model parameters (mu,Sigma) are Normal-Wishart distributed!
% end

% keyboard;
% filePath='C:\Users\manandhar\research\matlabStuff\Research\npbMil\vbGmmMilSynDataBags200TwoH1TwoH0R3';
% movie2gif(frameMovie,filePath);

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
    [N,D] = size(bags.data);
    KH = [KH1 KH0];

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
    if (myOptions.rndInit) 
        for mm=1:2
            eStep(mm).phi_l_i_m = rand(N,KH(mm)); 
            % Normalize
            eStep(mm).phi_l_i_m = eStep(mm).phi_l_i_m ./ repmat(sum(eStep(mm).phi_l_i_m,2),1,KH(mm));
        end
   % VB initialziation        
    else             
        % Initialize H0 respobsibility matrix (phi_l_i_0)
        % Cluster instances in B- bags
        [~,ctrsH0] = kmeans(bags.data(bags.label==0,:),myOptions.KH0,'EmptyAction','singleton','MaxIter',100);
        distMatrixH0 = pdist2(bags.data,ctrsH0);
        [~,indexKH0] = min(distMatrixH0,[],2);
        eStep(2).phi_l_i_m = bsxfun(@eq,indexKH0,1:myOptions.KH0);   % phi_l_i_0 before VB

        % Weight phi_l_i_0 by phi_l_0_M
        eStep(1).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(2).phi_l_i_m,eStep(3).phi_l_m_M(:,2));

        % Update H0 Mix Model (M-step for only H0 mixture model)    
        post(2) = npbMilMnUpdateThisMixMod(bags,prior(2),eStep(2));         

        % E-step only for H1 mixture model
        eStepThisMixMod = vbMilMnExpectationThisMixMod(bags,post(2),eStep(2),eStep(3).phi_l_m_M(:,2));

        % Compute log p(x_l|H0 MM); l={1,2,...,N_T}
        instancesLogPdf = npbMilLogSumLikelihood(bags,post,myOptions);

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
        [~,ctrsH1] = kmeans(bags.data(bags.myLabelH==1,:),myOptions.KH1,'EmptyAction','singleton');

        % Initialize phi_l_i_1
        distMatrixH1 = pdist2(bags.data,ctrsH1);
        [~,indexKH1] = min(distMatrixH1,[],2);
        eStep(1).phi_l_i_m = bsxfun(@eq,indexKH1,1:myOptions.KH1);   % phi_l_i_1

        % Initialize rH1H0 (big PHI)
        eStep(3).phi_l_m_M = zeros(N,2);
        eStep(3).phi_l_m_M(bags.myLabelH==1,1) = 1;
        eStep(3).phi_l_m_M(bags.myLabelH==0,2) = 1;

        % Weight phi_l_i_0 by phi_l_0_M
        eStep(1).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(1).phi_l_i_m,eStep(3).phi_l_m_M(:,1));

        % Update H1 Mix Model (M-step only for H1 mixture model)
        post(1) = npbMilMnUpdateThisMixMod(bags,prior(1),eStep(1));

        % E-step only for H1 mixture model
        eStepThisMixMod = vbMilMnExpectationThisMixMod(bags,post(1),eStep(1),eStep(3).phi_l_m_M(:,1));
        eStep(1).phi_l_i_m = eStepThisMixMod.phi_l_i_m;
    end    
    % Weight phi_l_i_0 by phi_l_0_M
    for mm=1:2
        eStep(mm).phi_l_m_M_phi_l_i_m = bsxfun(@times,eStep(mm).phi_l_i_m,eStep(3).phi_l_m_M(:,mm));
    end 
   
end

function post = npbMilMnUpdate(bags,prior,eStep,myOptions)
    for mm=1:2
        post(mm) = npbMilMnUpdateThisMixMod(bags,prior(mm),eStep(mm),myOptions);
    end
    % Eqn 24
    post(3).N_m_bar   = sum(eStep(3).phi_l_m_M(bags.label==1,:),1);   
    post(3).N_m_bar(post(3).N_m_bar==0) = eps;
    post(3).alpha_m  = post(3).N_m_bar + prior(3).alpha_m;
    post(3).eta_m = post(3).alpha_m/sum(post(3).alpha_m,2); 
end

function post = npbMilMnUpdateThisMixMod(bags,prior,eStep,myOptions)
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
    end  
end

function eStep = npbMilMnExpectation(bags,eStep,post,myOptions)
    % Eqn 108
    eStep(3).expect_log_eta_m    = psi(0,post(3).alpha_m) - psi(0,sum(post(3).alpha_m));
    for mm=1:2
        eStepThisMixMod = npbMilMnExpectationThisMixMod(bags,post(mm),eStep(mm),eStep(3).phi_l_m_M(:,mm),myOptions);
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

function eStep = npbMilMnExpectationThisMixMod(bags,post,eStep,rH1H0ThisMixMod,myOptions)
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

function nfeTerms = npbMilNegFreeEnergy(prior,post,Lav,sumKLZetaTerm,sumKLzTerm,sumExpectLLTerm,myOptions)
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

function [prior post] = npbMilPrune(prior,post,myOptions)
    for mm=1:2
        while any(post(mm).pi_i_m<myOptions.pruneThreshold)
            for k=1:length(post(mm).pi_i_m)
                if (post(mm).pi_i_m(k)<myOptions.pruneThreshold)   
                    prior(mm).meanMean(:,k)  = [];

                    post(mm).counts(k) = [];
                    post(mm).gamma1(k)  = [];
                    post(mm).gamma2(k)  = [];
                    post(mm).pi_i_m(k)      = [];
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

function [bags KH1 KH0 myOptions] = parseInputs(inputCell)

if (isstruct(inputCell{1}))
    bags = inputCell{1};
    switch length(inputCell)
        case 1 % vbGmmCluster(bags)
            KH1 = size(bags.data,1)/2;
            KH0 = size(bags.data,1)/2;
            myOptions.maxIteration = 5000;
            myOptions.convergenceThreshold = 1e-10;
            myOptions.pruneClusters = 1;
            myOptions.pruneThreshold = .02;
            myOptions.featsUncorrelated = 0;
            myOptions.rndInit = 1;
        case 2 % vbGmmCluster(bags,KH1)
            KH1 = inputCell{2};
            KH0 = inputCell{2};
            myOptions.maxIteration = 5000;
            myOptions.convergenceThreshold = 1e-10;
            myOptions.pruneClusters = 1;
            myOptions.pruneThreshold = .02;
            myOptions.featsUncorrelated = 0;
            myOptions.rndInit = 1;
        case 3 % vbGmmCluster(bags,KH1,KH0)
            KH1 = inputCell{2};
            KH0 = inputCell{3};
            myOptions.maxIteration = 5000;
            myOptions.convergenceThreshold = 1e-10;
            myOptions.pruneClusters = 1;
            myOptions.pruneThreshold = .02;
            myOptions.featsUncorrelated = 0;
            myOptions.rndInit = 1;
        case 4 % vbGmmCluster(bags,KH1,KH0,myOptions)
            KH1 = inputCell{2};
            KH0 = inputCell{3};
            if (isstruct(inputCell{4}))
                myOptions = inputCell{4};
                if (~isfield(myOptions,'maxIteration'))
                    myOptions.maxIteration = 5000;
                end
                if (~isfield(myOptions,'convergenceThreshold'))
                    myOptions.convergenceThreshold = 1e-10;
                end
                if (~isfield(myOptions,'pruneClusters'))
                    myOptions.pruneClusters = 1;
                end
                if (~isfield(myOptions,'pruneThreshold'))
                    myOptions.pruneThreshold = .02;
                end
                if (~isfield(myOptions,'featsUncorrelated'))
                    myOptions.featsUncorrelated = 0;
                end
                if (~isfield(myOptions,'rndInit'))
                    myOptions.rndInit = 1;
                end
            end
        otherwise
            error('Invalid number of input arguments');
    end
else
    error('Invalid input - bags should be a structure');
end
end

function logSumlikelihoodHMM = npbMilLogSumLikelihood(bags,post,myOptions)
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
    logSumlikelihoodHMM = log(likelihoodHMM);
end

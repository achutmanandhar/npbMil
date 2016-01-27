% PRTBRVMM - PRT BRV Mixture Model
%   A dirichlet density is used as the model for the mixing proportions. 
%   
%   Inherits from (prtBrv & prtBrvIVb & prtBrvVbOnline) and impliments all
%   required methods.
%
%   The construtor takes an array of prtBrvObsModel objects
%
%   obj = prtBrvMm(repmat(prtBrvMvn(2),3,1)); % A mixture with 3 2d MVNs
%
% Properties
%   mixing - prtBrvDiscrete object representing the dirichlet
%       density.
%   components - array of prtBrvObsModel components
%   nComponents - number of components in the mixture (Read only)
%
% Methods:
%   vb - Perform VB inference for the mixture
%   vbOnlineUpdate - Used within vbOnline() (Alpha release, be careful!)
%   vbNonStationaryUpdate - Performs one iteration of VB updating with
%       stabilized forgetting. (Alpha release, be careful!)

% Copyright (c) 2014 CoVar Applied Technologies
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the
% following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
% NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
% DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
% OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
% USE OR OTHER DEALINGS IN THE SOFTWARE.






classdef prtBrvMilMixture < prtBrv & prtBrvVbOnline & prtBrvMembershipModel
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Properties required by prtAction
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    properties (SetAccess = private)
        name = 'Mixture Bayesian Random Variable';
        nameAbbreviation = 'BRVMix';
    end
    
    properties (SetAccess = protected)
        isSupervised = false;
        isCrossValidateValid = true;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods for prtBrv
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
    methods
        function self = estimateParameters(self, x)
            self = conjugateUpdate(self, self, x);
        end
        
        function y = predictivePdf(self, x)
            y = exp(predictiveLogPdf(self, x));
        end
        function y = predictiveLogPdf(self, x)
            
            logLikelihoods = zeros(size(x,1), self.nComponents);
            for iComp = 1:self.nComponents
                logLikelihoods(:,iComp) = predictiveLogPdf(self.components(iComp),x);
            end
            
            piHat = self.mixing.posteriorMeanStruct;
            piHat = piHat.probabilities(:)';
            
            y = prtUtilSumExp(bsxfun(@plus,logLikelihoods,log(piHat))')';
            
        end
        
        function val = getNumDimensions(self)
            val = self.components(1).nDimensions;
        end
        
        function self = initialize(self, x)
            x = self.parseInputData(x);
            
            for iComp = 1:self.nComponents
                self.components(iComp) = self.components(iComp).initialize(x);
            end
            self.mixing = self.mixing.initialize(zeros(1,self.nComponents));
        end
        
        % Optional methods
        %------------------------------------------------------------------
        function val = plotLimits(self)
            allVal = zeros(self.nComponents, self.components(1).plotLimits);
            for s = 1:self.nComponents
                allVal(s,:) = obj.components(s).plotLimits();
            end
            val = zeros(1,size(allVal,2));
            for iDim = 1:size(allVal,2)
                if mod(iDim,2)
                    val(iDim) = min(allVal(:,iDim));
                else
                    val(iDim) = max(allVal(:,iDim));
                end
            end
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods for prtBrvVb
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    methods
        function [self, training] = vbBatch(self, x)
            [x, y] = self.parseInputData(x);

            self = initialize(self,x);
            
            % Initialize
            if self.vbVerboseText
                fprintf('\n\nVB inference for a mixture model with %d components\n', self.nComponents)
                fprintf('\tInitializing VB Mixture Model\n')
            end
            
            [self, prior, training] = vbInitialize(self, x, y);
            keyboard;
            
            if self.vbVerboseText
                fprintf('\tIterating VB Updates\n')
            end
                
            for iteration = 1:self.vbMaxIterations

                % VBM Step
                [self, training] = vbM(self, prior, x, y, training);
            
                % VBE Step
                [self, training] = vbE(self, prior, x, y, training);
                
%                 % Calculate NFE
%                 [nfe, eLogLikelihood, kld, kldDetails(iteration,1)] = vbNfe(self, prior, x, training);
                
                % Update training information
%                 training.previousNegativeFreeEnergy = training.negativeFreeEnergy;
%                 training.negativeFreeEnergy = nfe;
%                 training.iterations.negativeFreeEnergy(iteration) = nfe;
%                 training.iterations.eLogLikelihood(iteration) = eLogLikelihood;
%                 training.iterations.kld(iteration) = kld;
%                 training.nIterations = iteration;
%                 training.kld = kld;
%                 training.eLogLikelihood = eLogLikelihood;
                
                % Check covergence
%                 if self.vbCheckConvergence && iteration > 1
%                     [converged, err] = vbIsConverged(self, prior, x, training);
%                 else
%                     converged = false;
%                     err = false;
%                 end
                % 26/01/2016 AM - Ad-hoc
                if iteration>100
                    converged = true;
                    err = false;
                else
                    converged = false;
                    err = false;
                end
            
                % Plot
                if self.vbVerbosePlot && (mod(iteration-1,self.vbVerbosePlot) == 0)
                    % 26/01/2016 AM - Iterate over mixtures
                    for iMix=1:self.nComponents
                        figure(iMix);
                        vbIterationPlot(self.components(iMix), prior.components(iMix), x, training(iMix));
                    end
                    
                    if self.vbVerboseMovie
                        if isempty(self.vbVerboseMovieFrames)
                            self.vbVerboseMovieFrames = getframe(gcf);
                        else
                            self.vbVerboseMovieFrames(end+1) = getframe(gcf);
                        end
                    end
                end
%                 keyboard;
                
                if converged
                    if self.vbVerboseText
                        fprintf('\tConvergence reached. Change in negative free energy below threhsold.\n')
                    end
                    break
                end
                
                if err
                    % These might be useful things for debuging
                    
                    %eLogLikeDiff = training.eLogLikelihood - prevTraining.eLogLikelihood
                    %kldDiff = training.kld - prevTraining.kld
                    %mixingKldDiff = kldDetails(end).mixing - kldDetails(end-1).mixing
                    %componentKldDiff = sum(kldDetails(end).components) - sum(kldDetails(end-1).components)
                    %membershipKldDiff = sum(kldDetails(end).memberships)-sum(kldDetails(end-1).memberships)
                    %keyboard
                    
                    break
                end
                
                prevTraining = training;
                prevSelf = self;
                
            end
            if self.vbCheckConvergence && self.vbVerboseText
                fprintf('\nAll VB iterations complete.\n\n')
            end
            if self.vbCheckConvergence && ~converged && ~err && self.vbVerboseText
                fprintf('\nLearning did not complete in the allotted number of iterations.\n\n')
            end
            
            % 26/01/2016 AM - Commented for now
%             training.endTime = now;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods for prtBrvMcmc
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods for prtBrvVbMembershipModel
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % We don't actualy inherit from prtBrvVbMembershipModel yet so we don't
    % actually have to implement this but we do
    methods
        function y = conjugateVariationalAverageLogLikelihood(obj,x)
            
            training = prtBrvMilMixtureVbTraining;
            
            [twiddle, training] = obj.vbE(obj, x, training); %#ok<ASGLU>
            y = sum(prtUtilSumExp(training.variationalLogLikelihoodBySample'));
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Methods for prtBrvVbOnline
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    methods
        function [obj, priorObj, training] = vbOnlineInitialize(obj,x)
            
            training = prtBrvMixtureVbTraining;
            
            obj = initialize(obj, x);
            
            priorObj = obj;
            
            % Intialize mixing
            obj.mixing = obj.mixing.vbOnlineInitialize([]);
            
            % Initialize components
            obj.components = vbOnlineCollectionInitialize(obj.components, x);
            
        end
        
        function [obj, training] = vbOnlineUpdate(obj, priorObj, x, training, prevObj, learningRate, D)
            
            if nargin < 5 || isempty(prevObj)
                prevObj = obj;
            end
            
            if nargin < 4 || isempty(training)
                training = prtBrvMixtureVbTraining;
                training.iterations.negativeFreeEnergy = [];
                training.iterations.eLogLikelihood = [];
                training.iterations.kld = [];
                [obj, training] = obj.vbE(priorObj, x, training);
            end
            
            % Update components
            for s = 1:obj.nComponents
                obj.components(s) = obj.components(s).vbOnlineWeightedUpdate(priorObj.components(s), x, training.componentMemberships(:,s), learningRate, D, prevObj.components(s));
            end
            obj.mixing = obj.mixing.vbOnlineWeightedUpdate(priorObj.mixing, training.componentMemberships, [], learningRate, D, prevObj.mixing);
            
            training.nSamplesPerComponent = sum(training.componentMemberships,1);
            
            %[nfe, eLogLikelihood, kld, kldDetails] = vbNfe(obj, priorObj, x, training); %#ok<NASGU,ASGLU>
            %training.negativeFreeEnergy = -kld;
            
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Properties and Methods for prtBrvMixture use
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        function self = prtBrvMilMixture(varargin)
            if nargin < 1
                return
            end
            self = constructorInputParse(self,varargin{:});
        end
        
        % This could potentially be abstracted by prtBrvVbOnlineNonStationary but
        % that does not exist yet.
        function [obj, training] = vbNonStationaryUpdate(obj, priorObj, x, training, prevObj)
            
            if nargin < 5 || isempty(prevObj)
                prevObj = obj;
            end
            
            if nargin < 4 || isempty(training)
                training = prtBrvMixtureVbTraining;
                training.startTime = now;
                training.iterations.negativeFreeEnergy = [];
                training.iterations.eLogLikelihood = [];
                training.iterations.kld = [];
                [obj, training] = obj.vbE(priorObj, x, training);
            end
            
            obj.nSamples = obj.nSamples + size(x,1);
            obj.vbOnlineT = obj.nSamples;
                        
            % Update components
            for s = 1:obj.nComponents
                cBaseDensity = prevObj.components(s).weightedConjugateUpdate(prevObj.components(s),x,training.phiMat(:,s));
                obj.components(s) = obj.components(s).vbOnlineWeightedUpdate(priorObj.components(s), x, training.phiMat(:,s), obj.vbOnlineLambda, obj.vbOnlineD, cBaseDensity);
            end
            cBaseDensity = prevObj.mixing.weightedConjugateUpdate(prevObj.mixing,training.phiMat,[]);
            obj.mixing = obj.mixing.vbOnlineWeightedUpdate(priorObj.mixing, training.phiMat, [], obj.vbOnlineLambda, obj.vbOnlineD, cBaseDensity);

        end      
    end
    
    % Some properties (some extra hidden private properties to avoid
    % property access issues  when loading and saving
    %----------------------------------------------------------------------
    properties (Dependent) 
        mixing
        components
    end
    properties (Hidden, SetAccess='private', GetAccess='private');
        internalMixing = prtBrvDiscrete;
        internalComponents = prtBrvMvn;
    end
    properties (Dependent, SetAccess='private')
        nComponents
    end
    properties (Hidden)
        plotComponentProbabilityThreshold = 0.01;
    end
    
    % Set and get methods for weird properties
    %----------------------------------------------------------------------
    methods
        function obj = set.components(obj,components)
            assert( isa(components,'prtBrvMembershipModel'),'components must be a prtBrvMembershipModel')
            
            obj.internalComponents = components;
        end
        
        function val = get.components(obj)
            val = obj.internalComponents;
        end
        
        function obj = set.mixing(obj,mix)
            obj.internalMixing = mix;
        end
        
        function val = get.mixing(obj)
            val = obj.internalMixing;
        end
        
        function val = get.nComponents(obj)
            val = obj.getNumComponents();
        end
        
        function val = getNumComponents(self)
            val = length(self.components);
        end
    end
    
    
    % Methods for doing VB (called by batch VB above)
    %----------------------------------------------------------------------
    methods
        function [obj, priorObj, training] = vbInitialize(obj, x, y)
            % 26/01/2016 AM 
            % Initialize each of the two prtBrvMixture
            % training(1) = training(h1mix) also contains training(h1h0mix)
            % training(2) = training(h0mix)
            % training is overloaded. Should make hierarchical?
%             keyboard;
            for iMix=1:obj.nComponents
                training(iMix) = prtBrvMilMixtureVbTraining;
                
                priorObj.components(iMix) = obj.components(iMix);
                [training(iMix).componentMemberships, priorObj.components(iMix).components] = ...
                    collectionInitialize(obj.components(iMix).components, priorObj.components(iMix).components, x);

                training(iMix).variationalLogLikelihoodBySample = -inf(size(x,1),obj.components(iMix).nComponents);
            end
            
%             keyboard;
            % 26/01/2016 AM 
            % Initialize the overall prtBrvMilMixture
            rH1H0 = eps*ones(size(x,1),2);
            fuzzFactor = .1;
            rH1H0(y==0,1) = fuzzFactor*rand(sum(y==0),1); 
            rH1H0(y==0,2) = 1-eps;
            rH1H0 = bsxfun(@rdivide,rH1H0,sum(rH1H0,2)); % Normalize
            training(1).mixtureMemberships = rH1H0;
            training(1).variationalMixtureLogLikelihoodBySample = -inf(size(x,1),obj.nComponents);
            
            % 26/01/2016 AM
            % weight small phi's by big PHI
            % weight componentMemberships by mixtureMemberships
            for iMix=1:obj.nComponents
                training(iMix).componentMemberships = bsxfun(@times,training(iMix).componentMemberships,training(1).mixtureMemberships(:,iMix));
            end 
            
            % 26/01/2016 AM
            % Prior for the mixing of the two mixtures???
            % Gotta change this later!
            priorObj.mixing = obj.mixing;
            
        end
        
        function [obj, training] = vbE(obj, priorObj, x, y, training) %#ok<INUSL>
            % 26/01/2016 AM
            % Calculate the variational Log Likelihoods of each mixture
            for iMix=1:obj.nComponents
                % Calculate the variational Log Likelihoods of each cluster
                training(iMix).variationalClusterLogLikelihoods = zeros(size(x,1),obj.components(iMix).nComponents);
                for iSource = 1:obj.components(iMix).nComponents
                    training(iMix).variationalClusterLogLikelihoods(:,iSource) = ...
                        obj.components(iMix).components(iSource).conjugateVariationalAverageLogLikelihood(x);
                end
                
                % 26/01/2016 AM
                % In contrast to the eStep in standard prtBrvMvn,...
                % variationalClusterLogLikelihoods is weighted by mixtureMemberships just to calculate componentMemberships?
                variationalClusterLogLikelihoodsWeighted = bsxfun(@times,log(training(1).mixtureMemberships(:,iMix)),training(iMix).variationalClusterLogLikelihoods);

                expectedLogMixing = obj.components(iMix).mixing.expectedLogMean;

                training(iMix).variationalLogLikelihoodBySample = bsxfun(@plus,training(iMix).variationalClusterLogLikelihoods, expectedLogMixing(:)');
                % 26/01/2016 AM
                % In contrast to the eStep in standard prtBrvMvn,...
                % variationalLogLikelihoodBySample is weighted by mixtureMemberships just to calculate componentMemberships?
                variationalLogLikelihoodBySampleWeighted = bsxfun(@plus,variationalClusterLogLikelihoodsWeighted, expectedLogMixing(:)');
                training(iMix).componentMemberships = exp(bsxfun(@minus, variationalLogLikelihoodBySampleWeighted, prtUtilSumExp(variationalLogLikelihoodBySampleWeighted')'));
            end

            % 26/01/2016 AM
            % Calculate the variational mixture Log Likelihoods
            expectedLogMixing = obj.mixing.expectedLogMean;
            
            training(1).variationalMixtureLogLikelihoods = zeros(size(x,1),obj.nComponents);
            for iMix=1:obj.nComponents
                training(1).variationalMixtureLogLikelihoods(:,iMix) = sum(training(iMix).componentMemberships.*training(iMix).variationalClusterLogLikelihoods,2);
            end
            training(1).variationalMixtureLogLikelihoodBySample = bsxfun(@plus,training(1).variationalMixtureLogLikelihoods,expectedLogMixing(:)');
            training(1).mixtureMemberships = exp(bsxfun(@minus, training(1).variationalMixtureLogLikelihoodBySample, prtUtilSumExp(training(1).variationalMixtureLogLikelihoodBySample')'));
            
            % 26/01/2016 AM - Following is ad-hoc! Can we remove this?
            % Enforce correct responsibilities for instances in Y=0 bags 
            training(1).mixtureMemberships(y==0,1) = eps;    
            training(1).mixtureMemberships(y==0,2) = 1-eps; 
            % Make sure log(rH1H0) wont be infs
            training(1).mixtureMemberships(training(1).mixtureMemberships==0) = eps;
            training(1).mixtureMemberships(training(1).mixtureMemberships==1) = 1-eps;
            
            % weight small phi's by big PHI
            % weight componentMemberships by mixtureMemberships
            for iMix=1:obj.nComponents
                training(iMix).componentMemberships = bsxfun(@times,training(iMix).componentMemberships,training(1).mixtureMemberships(:,iMix));
            end 
        
        end
        
        function [obj, training] = vbM(obj, priorObj, x, y, training)
            % 26/01/2016 AM
            % Iterate through each mixture model
            for iMix = 1:obj.nComponents
                % Iterate through each source and update using the current memberships
                for iSource = 1:obj.components(iMix).nComponents
                    obj.components(iMix).components(iSource) = ...
                        obj.components(iMix).components(iSource).weightedConjugateUpdate(...
                        priorObj.components(iMix).components(iSource), x, training(iMix).componentMemberships(:,iSource));
                end
                
                % Only use the samples from the Y=1 bags to update the following
                training(iMix).nSamplesPerComponent = sum(training(iMix).componentMemberships(y==1,:),1);

                % Updated mixing
                obj.components(iMix).mixing = obj.components(iMix).mixing.conjugateUpdate(...
                    priorObj.components(iMix).mixing, training(iMix).nSamplesPerComponent);
            end
            
            % 26/01/2016
            % training is overloaded. Should make hierarchical? See above vbInitialize
            training(1).nSamplesPerMixture = sum(training(1).mixtureMemberships,1);
            
            % Updated mixture mixing
            obj.mixing = obj.mixing.conjugateUpdate(priorObj.mixing, training(1).nSamplesPerMixture);
            
        end
        
        function [nfe, eLogLikelihood, kld, kldDetails] = vbNfe(obj, priorObj, x, training) %#ok<INUSL>
            
            sourceKlds = zeros(obj.nComponents,1);
            for s = 1:obj.nComponents
                sourceKlds(s) = obj.components(s).conjugateKld(priorObj.components(s));
            end
            mixingKld = obj.mixing.conjugateKld(priorObj.mixing);
            
            entropyTerm = training.componentMemberships.*log(training.componentMemberships);
            entropyTerm(isnan(entropyTerm)) = 0;
            
            logPi = obj.mixing.expectedLogMean;
            membershipKlds = sum(entropyTerm,2)-sum(bsxfun(@times, training.componentMemberships,logPi(:)'),2);
            
            % When checking the negative free energy sometimes you want to
            % include the membership matrix as a variational parameter and
            % sometimes you do not. We allow both.
            if obj.vbNfeIncludeMemberships
                kld = sum(sourceKlds) + mixingKld + sum(membershipKlds);
                eLogLikelihood = sum(sum(training.variationalClusterLogLikelihoods.*training.componentMemberships,2));
            else
                kld = sum(sourceKlds) + mixingKld;
                eLogLikelihood = sum(prtUtilSumExp(training.variationalLogLikelihoodBySample'));
            end
            
            nfe = eLogLikelihood - kld;
            
            if nargout > 3
                kldDetails.components = sourceKlds(:);
                kldDetails.mixing = mixingKld;
                kldDetails.memberships = membershipKlds(:);
            end
        end
        
        function vbIterationPlot(obj, priorObj, x, training) %#ok<INUSL>
            
            colors = prtPlotUtilClassColors(obj.nComponents);
            
            set(gcf,'color',[1 1 1]);
            
            subplot(3,2,1)
            mixingPropPostMean = obj.mixing.posteriorMeanStruct;
            mixingPropPostMean = mixingPropPostMean.probabilities;
            
            [mixingPropPostMeanSorted, sortingInds] = sort(mixingPropPostMean,'descend');
            
            bar([mixingPropPostMeanSorted(:)'; nan(1,length(mixingPropPostMean(:)))])
            colormap(colors(sortingInds,:));
            ylim([0 1])
            xlim([0.5 1.5])
            set(gca,'XTick',[]);
            title('Source Probabilities');
            
            subplot(3,2,2)
            if ~isempty(training.iterations.negativeFreeEnergy)
                plot(training.iterations.negativeFreeEnergy,'k-')
                hold on
                plot(training.iterations.negativeFreeEnergy,'rx','markerSize',8)
                hold off
                xlim([0.5 length(training.iterations.negativeFreeEnergy)+0.5]);
            else
                plot(nan,nan)
                axis([0.5 1.5 0 1])
            end
            title('Convergence Criterion')
            xlabel('Iteration')

            subplot(3,1,2)

            componentsToPlot = mixingPropPostMean > obj.plotComponentProbabilityThreshold;
            if sum(componentsToPlot) > 0
                plotCollection(obj.components(componentsToPlot),colors(componentsToPlot,:));
            end
               
            subplot(3,1,3)
            if obj.nDimensions < 4
                [~, cY] = max(training.componentMemberships,[],2);
                if size(x,1) < obj.vbIterationPlotNumSamplesThreshold;
                    allHandles = plot(prtDataSetClass(x,cY));
                else
                    allHandles = plot(bootstrapByClass(prtDataSetClass(x,cY), ceil(obj.vbIterationPlotNumBootstrapSamples./obj.nComponents)));
                end
                
                uY = unique(cY);
                for s = 1:length(uY)
                    cColor = colors(uY(s),:);
                    set(allHandles(s),'MarkerFaceColor',cColor,'MarkerEdgeColor',prtPlotUtilLightenColors(cColor));
                end
                legend('off');
                
%                 plotLimits = [];
%                 for s = 1:obj.nComponents
%                     plotLimits(s,:) = obj.components(s).plotLimits();
%                 end
%                 plotLimits = plotLimits(componentsToPlot,:);
%                 if ~isempty(plotLimits)
%                     if obj.nDimensions == 1
%                         xlim([min(plotLimits(:,1)),max(plotLimits(:,2))]);
%                     elseif obj.nDimensions == 2
%                         axis([min(plotLimits(:,1)),max(plotLimits(:,2)),min(plotLimits(:,3)),max(plotLimits(:,4))]);
%                     end
%                 end
            
            else
                if size(training.componentMemberships,1) < obj.vbIterationPlotNumSamplesThreshold;
                    area(training.componentMemberships(:,sortingInds),'edgecolor','none')
                    % colormap set above in bar.
                else
                    cMemberships = training.componentMemberships(prtRvUtilRandomSample(size(training.componentMemberships,1),obj.vbIterationPlotNumBootstrapSamples),sortingInds);
                    area(cMemberships,'edgecolor','none');
                    % colormap set above in bar.
                end
                    ylim([0 1]);
                    title('Cluster Memberships');
            end
            
            drawnow;
        end
    end
    
    % Methods for prtBrvMembershipModel
    %----------------------------------------------------------------------
    methods 
        function [phiMat, priorVec] = collectionInitialize(selfVec, priorVec, x)
            phiMat = zeros(size(x,1),length(selfVec));
            randInd = prtRvUtilDiscreteRnd([1 2],[0.5 0.5],size(x,1));
            phiMat(sub2ind(size(phiMat), (1:size(x,1))',randInd)) = 1;
        end
        
        function self = weightedConjugateUpdate(self, prior, x, weights, training)
            
            % Iterate through each source and update using the current memberships
            for iSource = 1:self.nComponents
                self.components(iSource) = self.components(iSource).weightedConjugateUpdate(prior.components(iSource), x, weights.*training.componentMemberships(:,iSource));
            end
    
            training.nSamplesPerComponent = sum(bsxfun(@times,training.componentMemberships,weights),1);
            
            % Updated mixing
            self.mixing = self.mixing.conjugateUpdate(prior.mixing, training.nSamplesPerComponent);
        end
        
        function self = conjugateUpdate(self, prior, x) %#ok<INUSL>
            warning('prt:prtBrvMixture:conjugateUpdate','Model is not fully conjugate resorting to vb');
            self = vb(self, x);
        end
        
        function plotCollection(selfs,colors)
            
            for iComp = 1:length(selfs)
                hold on;
                mixingPropPostMean = selfs(iComp).mixing.posteriorMeanStruct;
                mixingPropPostMean = mixingPropPostMean.probabilities;
            
                cComponents = mixingPropPostMean > selfs(iComp).plotComponentProbabilityThreshold;
                if any(cComponents)
                    plotCollection(selfs(iComp).components(cComponents), repmat(colors(iComp,:),sum(cComponents),1));
                end
                
                if iComp == 1
                    axesLimits = repmat(axis,length(selfs),1);
                else
                    axesLimits(iComp,:) = axis;
                end
            end
            hold off;
            axis([min(axesLimits(:,1)), max(axesLimits(:,2)), min(axesLimits(:,3)), max(axesLimits(:,4))]);
            
            
        end
        
    end
    properties
        vbNfeIncludeMemberships = true;
        vbIterationPlotNumSamplesThreshold = 1000;
        vbIterationPlotNumBootstrapSamples = 200;
    end
    methods (Hidden)
        function [x, y] = parseInputData(self,x) %#ok<MANU>
            % 26/01/2016 AM - Potential for bug!
            if isnumeric(x) || islogical(x)
                return
            % 26/01/2016 AM
            % x is prtDataSetClassMultipleInstance
            elseif prtUtilIsSubClass(class(x),'prtDataSetClassMultipleInstance')
                y = x.getExpandedTargets();
                x = x.getExpandedData();
            else 
                error('prt:prtBrvMilMixture:parseInputData','prtBrvMilMixture requires a prtDataSetClassMultipleInstance or a numeric 2-D matrix');
            end
        end
    end    
end
        

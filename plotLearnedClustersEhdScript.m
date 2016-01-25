% bags = downSampleBags(bags,40);    
myOptions.pruneThreshold = .04;    
fig1 = figure(1);
d = size(bags.data,2);

    nGrid   = 200;
    tempMin = min(bags.data(:,1:2));
    xMin    = tempMin(1)-1;
    yMin    = tempMin(2)-1;
    tempMax = max(bags.data(:,1:2));
    xMax    = tempMax(1)+1;
    yMax    = tempMax(2)+1;

    xx = linspace(xMin,xMax,nGrid);
    yy = linspace(yMin,yMax,nGrid);
    [xGrid yGrid] = meshgrid(xx,yy);

    gridSpace2D = cat(2,xGrid(:),yGrid(:));

    MM = [1 0];
    for mm=1:2
        predictives = zeros(size(gridSpace2D,1),1);
        K = length(posteriors(mm).pi);
        for k=1:K
            if (posteriors(mm).pi(k)>myOptions.pruneThreshold)                    
                predictives = predictives + posteriors(mm).pi(k)*mvnpdf(gridSpace2D,posteriors(mm).meanN(1:2,k)',posteriors(mm).covN(1:2,1:2,k)); % predictive densities (Gaussian)                    
            end
    %         predictives = predictives + (posteriors(mm).lambda(k)/sumLambda)*pdfNonCentralT(gridSpace2D,posteriors(mm).rho(:,k)',posteriors(mm).Lambda(:,:,k),posteriors(mm).degreesOfFreedom(k)); % predictive densities (t)
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

        subplot(1,2,mm);
        handle(mm) = imagesc(xx,yy,gridSpace2DClusters);caxis([0 .035]);
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
            if (posteriors(mm).pi(k)>myOptions.pruneThreshold)                    
                myPlotMvnEllipse(posteriors(mm).meanN(1:2,k)',posteriors(mm).covN(1:2,1:2,k),1,[],'w',2);                    
            end
    %         plotMvnEllipse(posteriors(mm).rho(:,k)',posteriors(mm).Lambda(:,:,k)\eye(d),1);
            % annotate contours with pi's
        end
        title(['H',num2str(MM(mm)),' Clusters'],'FontWeight','b','FontSize',12);
        ylabel('PC 2','FontWeight','b','FontSize',12);
        xlabel('PC 1','FontWeight','b','FontSize',12);
        hold off;
    end
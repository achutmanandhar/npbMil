% ds = prtDataGenMultipleInstance;
rvH1 = prtRvMvn('mu',[2 2],'sigma',eye(2));
rvH0 = prtRvMvn('mu',[-2 -2],'sigma',eye(2));

nObservations = 100;
nInstPerBag = 10;

targets = nan(nObservations,1);
milStruct = struct;
for i = 1:nObservations
    x = rvH0.draw(nInstPerBag);
    targets(i,1) = 0;

    if ~mod(i,2)
        targets(i,1) = 1;
        x(1,:) = rvH1.draw(1);
    end
    milStruct(i,1).data = x;
end
ds = prtDataSetClassMultipleInstance(milStruct,targets);
bags.data = dsMil.expandedData;
bags.label = dsMil.expandedTargets;

h1mix = prtBrvMixture('components',repmat(prtBrvMvn,10,1), 'vbVerboseText',true, 'vbVerbosePlot', true, 'vbConvergenceThreshold',1e-10);
h0mix = prtBrvMixture('components',repmat(prtBrvMvn,10,1), 'vbVerboseText',true, 'vbVerbosePlot', true, 'vbConvergenceThreshold',1e-10);
mix = prtBrvMilMixture('components',[h1mix,h0mix]', 'vbVerboseText',true, 'vbVerbosePlot', true, 'vbConvergenceThreshold',1e-10);
[mixLearned, training] = mix.vbBatch(ds);
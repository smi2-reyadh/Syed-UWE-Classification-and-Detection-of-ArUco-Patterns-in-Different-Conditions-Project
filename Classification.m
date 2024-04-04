%% MATLAB code to recognise which pattern is present images in File 2

imds = imageDatastore('File2\basic\', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); % use folder names as class labels
    'ReadFcn';@(f) repmat(imresize(imread(f),input_layer_size(1:2)),[1,1,3]); % resize images to match that required by AlexNet

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');


numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

net = alexnet;

% analyzeNetwork(net)

%net.Layers(1).InputSize = [227,277,1];

inputSize = net.Layers(1).InputSize;

% inputSize = [227,277,1];

layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];


pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter,'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing', 'gray2rgb');

options = trainingOptions('sgdm', ... % use gradient decsent (with momentum)
    'MiniBatchSize',16, ... % number of images used per mini-batch
    'MaxEpochs',10, ... % maximum number of epochs to use in training
    'InitialLearnRate',5e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',1000, ...
    'Plots','training-progress', ...
    'WorkerLoad',1); % plot progress during training


netTransfer = trainNetwork(augimdsTrain,layers,options);

[YPred,scores] = classify(netTransfer,augimdsValidation);


idx = randperm(numel(imdsValidation.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end


YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation);

%% Confusion matrix
[cmap,clabel] = confusionmat(imdsValidation.Labels,YPred); % calculate confusion matrix
heatmap(clabel,clabel,cmap); % draw confusion matrix








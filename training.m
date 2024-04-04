% Sample script to demonstrate training a detection network. 
clear
close all
clc
rng(0)

dataFileName = '..\combinedPicsBasic\BBData.mat';

% load image data table
load(dataFileName)

% split into training + test
indexTrainEnd = floor(0.7*height(labelsGroundTruth));
indexTestStart = indexTrainEnd + 1 + floor(0.1*height(labelsGroundTruth));
trainingDataTable = labelsGroundTruth(1:indexTrainEnd,:);
validationDataTable = labelsGroundTruth(indexTrainEnd+1:indexTestStart-1,:);
testDataTable = labelsGroundTruth(indexTestStart:end,:);

% get bounding boxes and imagestore into correct format for training then combine
imdsTrain = imageDatastore(trainingDataTable.fileNames); % file name column (image datastore)
bxdsTrain = boxLabelDatastore(trainingDataTable(:,2:end)); % bounding box columns (bounding box datastore)
trainingData = combine(imdsTrain,bxdsTrain);

% repeat for validation
imdsValidation = imageDatastore(validationDataTable.fileNames);
bxdsValidation = boxLabelDatastore(validationDataTable(:,2:end));
validationData = combine(imdsValidation,bxdsValidation);

% repeat for testing
imdsTest = imageDatastore(testDataTable.fileNames);
bxdsTest = boxLabelDatastore(testDataTable(:,2:end));
testData = combine(imdsTest,bxdsTest);

% estimate anchor box sizes (one for each class for now)
anchorBoxes = estimateAnchorBoxes(trainingData,1);

% load pretrained network
imageSize = [imfinfo(imdsTrain.Files{1}).Height imfinfo(imdsTrain.Files{1}).Width 3];
lgraph = yolov2Layers([224 224 3],1,anchorBoxes,resnet18,"res5b_relu");

% set options
options = trainingOptions('adam',...
          'InitialLearnRate',0.002,...
          'Verbose',true,...
          'MiniBatchSize',8,... % number of images used per mini-batch
          'MaxEpochs',50,... % maximum number of epochs to use in training
          'Shuffle','every-epoch',...
          'VerboseFrequency',20,...
          'ValidationFrequency',50,...
          'CheckpointPath',tempdir,...
          'Plots','training-progress',...
          'ValidationData',validationData,...
          'ExecutionEnvironment','gpu'); % plot progress during training


% do training
detector = trainYOLOv2ObjectDetector(trainingData,lgraph,options);

save('trainedNetwork','detector','testDataTable','testData') 



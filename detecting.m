% Script to demonstrate training and testing a detection network.
clear
close all
clc
load trainedNetwork

imExample = imread(testDataTable.fileNames{1});
[dbox,dscore,dlabel] = detect(detector,imExample);
figure
imshow(insertObjectAnnotation(imExample,"rectangle",dbox,dlabel))

% show all results on test images
close
for imNo = 1:height(testDataTable)
    imExample = imread(testDataTable.fileNames{imNo});
    [dbox,dscore,dlabel] = detect(detector,imExample);
    subplot(2,4,imNo)
    imshow(insertObjectAnnotation(imExample,"rectangle",dbox,dlabel))
end

% overall precision
resultsFull = detect(detector, testData);
[averagePrecision,recall,precision] = evaluateDetectionPrecision(resultsFull, testDataTable(:,2:end));
sprintf('Average precision = %.3f\n',averagePrecision)

% options recall-precision graph
% Good source: https://medium.com/axinc-ai/map-evaluation-metric-of-object-detection-model-dd20e2dc2472
figure
subplot(221)
plot(recall{1},precision{1},'.-')
xlabel('Recall');ylabel('Precision')
axis([0 1 0 1])
title(sprintf('"0": Average precision = %.2f\n',averagePrecision(1)))
subplot(222)
plot(recall{2},precision{2},'.-')
xlabel('Recall');ylabel('Precision')
axis([0 1 0 1])
title(sprintf('"1": Average precision = %.2f\n',averagePrecision(2)))
subplot(223)
plot(recall{3},precision{3},'.-')
xlabel('Recall');ylabel('Precision')
axis([0 1 0 1])
title(sprintf('"2": Average precision = %.2f\n',averagePrecision(3)))
subplot(224)
plot(recall{4},precision{4},'.-')
xlabel('Recall');ylabel('Precision')
title(sprintf('"3": Average precision = %.2f\n',averagePrecision(4)))
axis([0 1 0 1])

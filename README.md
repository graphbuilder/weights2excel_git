# weights2excel
weights2excel

## Contents
For the weights (yolov4.best.0 & yolov4.cfg, mAP 82.99%, IOU 76.08%, Sparsity 93.06%)have been pruned, 
e.g. layer157 [1024, 512, 3, 3], set step as 16 thin batches, groups = 1024/16 = 64 as cols number, 
and it has 512 channels as row's number.

Than calculate the unit(row, col)'s max value nozero number,
and calculate the max values BC in all batches for every channels.
Last print total max values in all BC and the sparsity this layers, by the way.

So, by EXCEL, we can draw the BCs as a bar, summary as the nozero weights after prune
every channel still obey Normal distribution．

from sys import path as syspath
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/")
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/")
syspath.append("/home/asalerno/Documents/pyDirectionCompSense/source/wavelet")
from waveletExampleInputs import runCSAlgorithm

pctg = [0.125, 0.25, 0.33, 0.50, 0.75]

saveImsPngFile = []
saveImDiffPngFile = []
saveNpyFile = []

for i in xrange(len(pctg)):
    saveImsPngFile.append('/micehome/asalerno/Documents/pyDirectionCompSense/waveletTests/'+str(int(pctg[i]*100))+'_spConvergences')
    saveNpyFile.append('/micehome/asalerno/Documents/pyDirectionCompSense/waveletTests/'+str(int(pctg[i]*100))+'_spConvergences.npy')
    saveImDiffPngFile.append('/micehome/asalerno/Documents/pyDirectionCompSense/waveletTests/'+str(int(pctg[i]*100))+'_spDifferences')

for i in xrange(len(pctg)):
    runCSAlgorithm(pctg=pctg[i],
                   saveNpy=True,
                   saveNpyFile=saveNpyFile[i],
                   saveImsPng=True,
                   saveImsPngFile=saveImsPngFile[i],
                   saveImDiffPng=True,
                   saveImDiffPngFile=saveImDiffPngFile[i])
# AccelCluster-Visualizations

## Order to run:
- stitched_labeled_experiment.ipynb
  - Input: session_1_out.m
  - Output: combined_matrix_final.csv
- circularHeatmap.ipynb
  - Input: combined_matrix_final.csv
  - Output: eps and png graphs
- similarityMatrix.ipynb
  - Input: session_1_out.mat
  - Output: groupedClusters.csv, eps and png graphs
- sankey.ipynb
  - Input: combined_matrix_final.csv, groupedClusters.csv
  - Output: svg and png graphs
- transitionProbabilityMatrixArena.ipynb
  - Input: combined_matrix_final.csv and groupedClusters.csv
  - Output: svg and png graphs
- tSNE.ipynb
  - Input: combined_matrix_final.csv
  - Output: svg and png graphs

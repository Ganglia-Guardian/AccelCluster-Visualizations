Currently, the social box data is being taken from the 'SB_Data' folder and all social box visualizations are being outputted to the 'SB_Visualizations' folder.
Similarly, the different tier arena data is being taken from the 'Arena_Data' folder and all arena visualizations are being outputted to the 'Arena_Visualizations' folder.
Feel free to add data as you receive it - be aware that you will need to change the 'data_dir' and 'visualizations_dir' in each of the files to the correct directory. 
Each visualization will be saved in .png and .eps format, if possible. The social box visualizations are separated into full_data, open_field, social_box, roi_male,
roi_female, roi_bully, roi_dummy. The arena visualizations are separated into full_data, arena1_data, arena2_data, arena3_data, arena4_data, and arena5_data. 

All data is processed through the AccelCluster Matlab pipline. From those results, the relevant files are the 'Clusters' file (might be within session_out.mat). From the
'Clusters' file, we use the 'sim' file which contains the similarity matrix and the 'idx' file while contains all of the cluster assignments, in order. Assume this is taken
in 300 ms intervals. 

SOCIAL BOX 

The full data is as given from the AccelCluster pipeline. The social box and open field data needs to be split manually. The AccelCluster pipeline will stitch together
the open field and social box data - disucss with people that are in charge of the pipeline to figure out the exact place where the data switches from one to the other 
and split at that time to create a social box and open field data frame. Each of the individual region of interest (ROI) timestamps are taken from the 'tdt fiber photometry' 
pipeline. In Teams,look in tdt fiber photometry --> OF vs SB --> Timestamp Extraction.ipynb. This notebook should contain all the instructions to extract the timestamps for 
each individual ROI. All of these files should make up the whole of the data needed to run all files regarding social box. 

ARENA 

The arena data is much simpler to get into order. All five arenas will be stitched together. The current code assumes that each arena has 1 hour of corresponding cluster
assignments and that the arenas are stitched together in order ( small open field/restricted, large open field/open, low tier, mid tier, high tier). However, both of these
must be double checked with whoever is running the AccelCluster pipeline. If it less/more than 1 hour per arena, those calculations need to be adjusted at the start
of each notebook producing arena visualizations. If they are stitched out of order, then the labels may also need to be changed based on the arena number it is 
corresponding to (for example, arena 1 might indicate mid tier arena when we previously assumed it was small open field). This is why clarifying the order and duration
of each arena is important to do ahead of generating the visualizations. 

SIMILARITY MATRIX 

** The similarity matrix actually contains negative values, making it a dissimilarity matrix. 0 indicates the highest amount of similarity. **

This notebook contains all the code to generate various similarity matrices as well as dendrograms. The original similarity matrix generated from Matlab contains input
for every timestamp. Each timestamp correlates to a specific cluster assignment. This notebook reorders the the timestamps in order of their corresponding cluster value
and then averages together all rows and columns where the cluster assignment is the same. This creates the final similarity matrix of dimensions (number of clusters) x 
(number of clusters). There is also then a function to do a similar process with the relevant rows and columns of the similarity matrix with the broken down arenas or 
regions of interest. Then, the dendrogram is created using the final similarity matrix. The built in linkage() function is used to create a linkage matrix that contains
the distance between each of the clusters. Right now, the method used to create the linkage matrix is 'average'. The linkage() functions performs hierarchial clustering. 
This information is then plotted in the form of a dendrogram. That dendrogram is then used to figure out which clusters can be grouped together. For example, one session
of the arena data produced 74 unique clusters, so having some of them grouped together would make the analysis more meaningful. These groups are assigned by figuring out
a threshold on the dendrogram to where any clusters that are grouped together under the threshold are grouped together in the final groups. The threshold is assigned
manually. This creates uneven groups, but is the best method we have found until now. Feel free to explore more.

TRANSITION PROBABILITY 

This notebook contains all the code to generate transition probability matrices, transition probability graphs, and the top 10 transition probability graphs. Right now,
the transitions are manually counted and stored in a dictionary. This is because - specifically in social box - with the way that the regions of interests are split, it 
may happen that there appears to be a transition when strictly looking at that region of interest, but that transition does not hold when looking at all of the data 
sequentially. So, as of now the code verifies that any transition in a subset of the data is also a transition in the full data and then stores it in a dictionary. That
is then first plotted as a heatmap. Then, the same information is plotted as a transition graph. The code uses a circular layout for the nodes which represent the clusters.
The larger the size of the node, the more transitions that it is involved in. The edges represent that a transition occurs between two clusters. The width of the edge
represents the strength of the probability that that transition will occur. It is a little overwhelming because it plots all the transitions that occur. To help this
issue, we also isolate and plot the top 10 transitions with the highest probabilites. The final plots just show the top 10 transitions. The groups extracted from the 
dendrogram are also put into use for this graph - clusters are colored based on what group they are part of. This allows you to see the most meaningful transitions. 

CIRCULAR HEATMAP 

This notebook generates a circular heatmap indicating the change of usage. In social box, all regions of interest are compared to dummy. So, we are seeing how the 
cluster usage increases/decreases relative to the dummy region of interest. In the arena data, all arenas are compared the small open field/restricted arena. We are 
assuming that this is arena 1 - but again, double check that this is correct. It also then generates a heatmap for the change of usage for the groups as a whole. Ideally,
we will be able to produce one visual that has two tracks - the outer track showing the change in cluster usage and the inner track showing the change in group usage. Try
using the Circos library to create this visualization.

CLUSTER USAGE PI CHARTS

This notebook generates cluster usage pi charts. For Social Box, the pi charts are split up by region of interest and for Arenas, its split up between the 5 arenas. Only
the idx file which is generated by the matlab pipeline is used for this code. 

SANKEY

The sankey code uses the groupings generated by the dendrogram and the idx file to plot every single transition between clusters. Pi charts are also created to show the
incoming and outgoing transitions for each sankey. The sankey plots are split up by ROI for SB and by arenas for 3D arenas. 

USAGE MAP

The usage map contains the usage (by percentage) for each cluster split up by arena for the 3D arenas and by ROI for SB. The file also contains the change in usage between arenas
(comparing every arena with arena 1) and is normalized. The larger change in usage grpah is then split up into comparisons to make it easier to read. 

UMAP/TSNE

** Still in progress --> in 'Notebooks' folder: tsne_arena, tsne_SB ** (Sanjana's Computer)

These are both forms of dimensionality reduction to visualize higher dimension data in a 2D or 3D space. These notebooks are still in process as we have not figured
out the best way to execute this. Currently the idea is that we use the sensor data (given from the AccelCluster output) as the data itself and the assigned cluster
labels to color the data points. Sensor data is across 3 sensors and each sensor has 3 axes, giving you a total of data in 9 dimensions. The sensor data will be 
located in 'pairedextractedacced.mat' --> 'accxyz'/'gyroxyz'/'magxyz'. It does not seem that the output is what we expect to be right now. I think it's worth playing 
around with the different hyperparameters, specifically the n_neighbors, as well as maybe just plotting subsets of the original data. Dr. Gu will provide guidance on
the expected output and also reach to Jonathan Tang (Jonathan.Tang@seattlechildrens.org) for any clarifications. 

ADDITIONAL NOTES 

- Take a look around to see how each data file is structured. 
- Make sure you are running all of your code in a virtual environment. Instructions are located in Teams : 'AccelCluster' --> 'Visualization' --> 'Nivedha+Sanjana' --> Cleaned Visulaization Code ---> 'README.docx' (Umap + Tsne)
- Visualization Pipeline: AccelCluster -> Visualization -> Visualization Protocol -> AccelCluster


- A lot of the data used for visualizations does not correlate with timestamps. Often, you have to add them manually (in 300 ms intervals), or the timestamps don't add 
much to the analysis.
- Some plots take a long time when trying to save in .png/.eps format. If you want it to run faster, skip saving and manually save afterwards.
- There are a lot of Python functions used. They can all be tuned, if you wish.
- All paths to directories or files should be relative. Try to avoid using full data paths unless it's for testing purposes.


** Order of execution: similarity matrix -> transition probability -> circular heatmap -> cluster usage pi chart -> sankey -> usage map**

Please contact us if you have any questions! 

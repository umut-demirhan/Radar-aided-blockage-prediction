# Radar Aided Proactive Blockage Prediction in Real-World Millimeter Wave Systems
This is a python code package related to the following article:
Umut Demirhan, and Ahmed Alkhateeb, "[Radar Aided Proactive Blockage Prediction in Real-World Millimeter Wave Systems](https://arxiv.org/abs/2111.14805)," in 2022 IEEE International Conference on Communications (ICC), 2022

# Abstract of the Article
Millimeter wave (mmWave) and sub-terahertz communication systems rely mainly on line-of-sight (LOS) links between the transmitters and receivers. The sensitivity of these high-frequency LOS links to blockages, however, challenges the reliability and latency requirements of these communication networks. In this paper, we propose to utilize radar sensors to provide sensing information about the surrounding environment and moving objects, and leverage this information to proactively predict future link blockages before they happen. This is motivated by the low cost of the radar sensors, their ability to efficiently capture important features such as the range, angle, velocity of the moving scatterers (candidate blockages), and their capability to capture radar frames at relatively high speed. We formulate the radar-aided proactive blockage prediction problem and develop two solutions for this problem based on classical radar object tracking and deep neural networks. The two solutions are designed to leverage domain knowledge and the understanding of the blockage prediction problem. To accurately evaluate the proposed solutions, we build a large-scale real-world dataset, based on the DeepSense framework, gathering co-existing radar and mmWave communication measurements of more than 10 thousand data points and various blockage objects (vehicles, bikes, humans, etc.). The evaluation results, based on this dataset, show that the proposed approaches can predict future blockages 1 second before they happen with more than 90% F1 score (and more than 90% accuracy). These results, among others, highlight a promising solution for blockage prediction and reliability enhancement in future wireless mmWave and terahertz communication systems.

# Code Package Content 
The scripts for generating the results of the solution in the paper. This script adopts Scenario 30 of DeepSense6G dataset.

**To reproduce the results, please follow these steps:**
1. Download [the radar aided blockage prediction dataset of DeepSense 6G/Scenario 30](https://deepsense6g.net/radar-aided-blockage-prediction/).
2. Download (or clone) the repository into a directory.
3. Extract the dataset into the repository directory 
   (If needed, the dataset directory can be changed at line 21 of scenario30_radar_blockage-prediction_inference.py)
4. Run scenario30_radar_blockage-prediction_inference.py file.

If you have any questions regarding the code and used dataset, please write to DeepSense 6G dataset forum https://deepsense6g.net/forum/ or contact [Umut Demirhan](mailto:udemirhan@asu.edu?subject=[GitHub]%20Radar%20blockage%20prediction%20implementation).

# License and Referencing
This code package is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/). 
If you in any way use this code for research that results in publications, please cite our original article:
> U. Demirhan and A. Alkhateeb, "[Radar Aided Proactive Blockage Prediction in Real-World Millimeter Wave Systems](https://arxiv.org/abs/2111.14805)," in 2022 IEEE International Conference on Communications (ICC), 2022

If you use the [DeepSense 6G dataset](www.deepsense6g.net), please also cite our dataset article:
> A. Alkhateeb, G. Charan, T. Osman, A. Hredzak, and N. Srinivas, “DeepSense 6G: large-scale real-world multi-modal sensing and communication datasets,” to be available on arXiv, 2022. [Online]. Available: https://www.DeepSense6G.net

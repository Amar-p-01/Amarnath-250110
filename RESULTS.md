# DSB2018

I tried to explore different models for implementation 
=>Unet was asked to do 
=>Stardist 
=>Splinedist (basically in stardist the shape of object detected in star boundary but here the shape is flexible spline...so maP increase 
)
=>Cellpose , it gace the highest maP score 

* stardist, splinedist , cellpose are specifically trained for the nuclie detection

# Got these from research papers and tried to implement them using Repos


Stardist Training for stage gives maP of 58. somethin , while the test data dropped to 43 maP somethin 
IoU threshold 
IoU>0.5=0.695, IoU>0.75=0.451,IoU>0.9=0.109,IoU>0.95=0.005

Then with Cellpose-SAM
Cellpose 0.793 , 0.473 , 0.153 , 0.024  IoU scale 
will commit after training nd testing  on kaggle

Splinedist: unable to train and test but , but it is better than the stardist (verfied by data , which is test on the same dataset )
source:https://ieeexplore.ieee.org/document/9433928



## Few Findings

 All four methods share a Unet; they differ only in the output head and how instances are decoded
StarDist scored 0.581 on validation but 0.434 on test.
Although for splinedist this was written ""performance does not significantly differ, since most objects have nearly star-convex shapes" but unable to compare it with spline

Tried the Unet +  Watershed , watershed


Adding the direct code from cell of these different models 


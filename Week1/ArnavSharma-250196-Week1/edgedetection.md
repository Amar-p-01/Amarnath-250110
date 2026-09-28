
# Basic image processing:

I first just implemented Canny edge detection using opencv2 , pretty good results, it couldnt pick up the edges of the clouds very well but the building were solid. I tried chanding the size of kernel in Gausian blur as i thought that might help but there was no significant change.

Then i added a new function which does bilateralfiltering , by adjusting the values of d , simgacolor and sigmaspace,  i was able to remove the edges due to the cloud , so if we don't need the clouds we can do that.Paired with thresholding , it could be good.

Then at the end i just used canny which generated the best results , i had to adjust the threshold values a bit to include more details. Probably if paired with thresholding , bilateralfiltering could work better but i don't have time to do that for now.
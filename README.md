# hull_tactical
Prediction model for S&amp;P500 

This is a simple long short-term memory network which can train on data defined by the ucsbdata.csv. 
There are currently two models. one that can validate or predict on either the cpu or the gpu. and another which can train and validate on the gpu. 

If you do not have cuda, use cpu_model.py. You may use the pretrained model marked model.pth.
action: python3 cpu_model.py --load --data \<path to data\> 

The data being used must be in specific format of 66 features in the order specified in the ucsbdata.csv. 

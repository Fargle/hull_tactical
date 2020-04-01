# S&P Predictive Model
Prediction model for S&amp;P500 

# About
This is a simple long short-term memory (lstm) network which can train on data defined by the ucsbdata.csv. 
The model currently supports gpu and cpu machines. This project is made for the erp contest, a stock prediction contest. 

# Running the Code
If you would like to run this model for yourself you'll need these things before hand.
Everything you'll need to run this is contained in the requirements.txt file. 

# Example Usage
python3 model.py --train --parameters params.json
* you can mess with the params.json file and see if anything better works for you. 
* If you want to use different data you can specify with the --data <file path> command. 
python3 model.py --train --data <path to data> 
* After training, the model is written to a file model.pth. 
* To load a pretrained model
python3 model.py --load

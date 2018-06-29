# Introduction

The implementation is based on this paper; (https://research.fb.com/downloads/babi/):
Hybrid Code Networks: practical and efficient end-to-end dialog control with supervised and reinforcement learning
Jason D. Williams, Kavosh Asadi, Geoffrey Zweig
https://arxiv.org/abs/1702.03274

This diagram summarises the code and process used for the pharmabot

![alt text](https://github.com/fellowship/dialog/blob/master/pharmabot/Diagram%20for%20the%20pharmabot%20implementation%20of%20HCN.png)

For more detailed explanation go to: https://github.com/fellowship/dialog/blob/master/working_dev_only/PharmaHCN.ipynb

The code itself is forked from this repo https://github.com/jojonki/Hybrid-Code-Networks and has been adapted to use a new dataset. The dataset construction follows a similar process to the one used for bAbi task 5. 

For the moment, the data is generated randomly and then stored in a SQL database, to allow scaling to a much larger dataset. Training and test sets of 1000 dialogues were generated using the same code (the test set could be made harder by holding out particular phrasings of questions).

The data uses information scraped from the dailymed website (https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm) (unfortunately it is hard to get Novartis data from this website)

The dialog is stored as follows: 
good morning<\t>hello, what can I help you with today?<\n>
what do I do with drug56?<\t>what version of drug56 is that, for example: oral or injectable?<\n>
intravenous<\t>ok, let me look in to that<\n>
actually i meant subcutaneous<\t>ok, is there anything else to update?<\n>
actually i meant drug41<\t>ok, is there anything else to update?<\n>
no<\t>answer q3: instructions for drug 41_subcutaneous<\n>
thank you<\t>you're welcome<\n><\n>

###### Currently the questions will be better understood if asked in the following formats (given that we do not have real dialouges for Novartis drugs to train with): (defined in [novartis_config.py](https://github.com/fellowship/dialog/blob/master/pharmabot/novartis_config.py) file)
Q1: What is <drug_name>
Q2: Should I take <drug_name>, Is <drug_name> safe, can i take <drug_name>, 
Q4: how do I use <drug_name>, what do I do with <drug_name>, how do I take <drug_name>
Q5: how many <drug_name> is the correct dose, what is the dose for <drug_name>, how much <drug_name> should i take
Q6: what are the side effects of <drug_name>, is it normal to have <symptom> with <drug_name>, does <drug_name> cause <symptom>, is <symptom> a common side effect of <drug_name>, 
Q7: how do I store <drug_name>, where should I keep <drug_name>, do I need to keep <drug_name> in the dark


# Requirements:
- See [requirements.txt](https://github.com/fellowship/dialog/blob/master/pharmabot/requirements.txt)
- Download: Google News word2vec from: (https://code.google.com/archive/p/word2vec/)
- flask (to run Flask app download Flask>=0.11 and maybe SQLAlchemy>=1.1.11)
- DB browswer (for opening SQLite files)

# Running

To run the model, after installing the requirements:
- run [bot.py](https://github.com/fellowship/dialog/blob/master/pharmabot/bot.py)
- Then wait for the model to run and click the link:  http://0.0.0.0:8080/ in the console to access the Flask app.
- ask the bot the questions in the format that are similar to the ones described above.
- refresh the browser for a new conversation
- Note: if problems occur try the following:
  + Reload the pre_embd_w_full.pkl file (You will first need to change the file name, create an empty new file with same name then run bot.py)
  + Change the path for the Google News word2vec to your local path

# Further developments

- Scrape more Novartis related information from the pdfs on the Novartis website: (https://www.pharma.us.novartis.com/product-list)and replace the current [dailymed data file](https://github.com/fellowship/dialog/blob/master/pharmabot/dailymed_results_prescription.json)
- Implement the files faqmodel.bin to replace the Google News dataset and use faqmodel.pkl directly in the model (adapt the code in bot.py - first you will need to modify the args.vocab_full default to and the emb_size to 50 and use the commented out version of the bow variable instead of the current one).
  + the faqmodel.bin us obtained by generating a gensim model using 1.6 million sentences scraped from https://www.drugs.com/answers/questions/
- Add more questions to [novartis_config.py](https://github.com/fellowship/dialog/blob/master/pharmabot/novartis_config.py), currently there are only 6 type of questions.
- Add new questions to create a new dialogues_test.txt file to make the test less biased
- Improve the bot to have longer and more conversations
- Find real dialouges to train the model (We could not find real dialouges that are suitable for training but possible resources are: https://www.reddit.com/r/askmedicine/, https://www.drugs.com/, http://www.thepatientforum.com/default.aspx?PageID=5285)

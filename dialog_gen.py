# generates synthetic dialog data using input from config file and saves to text file

import sqlite3, random, re, copy, pickle
from utils import sub_reqs, get_entities

# change this to the relevant config file
from novartis_config import questions, optional_info, additional_qs, req_sub_dict
from synthetic_pharma_data import get_settings


#options
num_dialogues = 1000
database = 'NovartisQA'
set = 'test'
output = 'dialogues_'


def answer_quest(curr_dialogue, query, context_details):
    query = sub_reqs(cur, query, req_sub_dict, context_details)
    cur.execute(query)
    try:
        exec(globals()['answer'], globals())
        curr_dialogue += [str(question_num + len(action_dict) - len(questions) - 1) + ' ' + globals()['answer_text']]
    except:
        curr_dialogue += [str(question_num + len(action_dict) - len(questions) - 1) + ' '
                       + 'sorry, there is no information for that']
    return curr_dialogue


# load data from config file
questions = [q for q in questions.values()]
entity_dict = get_entities('entities.txt')
context_keys = sorted(entity_dict.keys())

''' 
create action dictionary
action#: [query, answer, [req_context], [blocking_context]]
context: {'age_groups': [], 'conditions': [], 'delivery_methods': [], 'drugs': [],
          'periods': [], 'severities': [], 'strengths': [], 'symptoms': [], 'units': []}
'''

action_dict = {
    0: ['''data = '<SILENT>' ''', 'answer_text = data', [], []],
    1: ['''data = 'hello, what can I help you with today?' ''', 'answer_text = data', [], []],
    2: ['''data = 'ok, is there anything else to update?' ''', 'answer_text = data', [], []],
    3: ['''data = 'ok, let me look in to that' ''', 'answer_text = data', [], []],
    4: ['''data = 'no problem' ''', 'answer_text = data', [], []],
}

# add the additional questions
act_size, basic_act_size = [len(action_dict)] *2
for requirement, question in additional_qs.items():
    action_dict[act_size] = ['data = \'' + question + '\'', 'answer_text = data', [], [context_keys.index(requirement)]]
    act_size += 1

# add the answers
for idx, question in enumerate(questions):
    query = question[4]
    answer = question[5]
    requirements = question[6].split(',')
    requirements = [context_keys.index(r) for r in requirements]
    action_dict[idx + act_size] = ['cur.execute("' + re.sub(r'\n','',query) + '")', answer, requirements, []]

#save dictionaries for use during training
pickle.dump(action_dict, open('actions.pkl', mode='wb'))
pickle.dump(req_sub_dict, open('req_sub_dict.pkl', mode='wb'))

# connect to database
conn = sqlite3.connect(database+'.sqlite')
cur = conn.cursor()

# get list of id for questions from table
# cur.execute('SELECT id FROM Questions')
# question_ids = cur.fetchall()

# generate synthetic dialogue data
dialogues = []
for d in range(num_dialogues):
    curr_dialogue = []

    # select a question and format
    # question_num = random.randint(1, len(question_ids))
    question_num = random.randint(0, len(questions))
    # cur.execute('SELECT * FROM Questions WHERE id = ?', (question_num, ))
    # question_data = cur.fetchall()
    question_data = questions[question_num]
    question_formats = question_data[0:4]
    query = question_data[4]
    answer = question_data[5]
    required_fields = question_data[6].split(',')
    new_formats = []
    for q_format in list(question_formats):
        # ignore empty strings
        if q_format:
            new_formats += [q_format]
    question_formats = list(new_formats)
    question = question_formats[random.randint(0, len(question_formats)-1)] + '?'

    # generate required fields from available options
    context_details = get_settings()

    # maybe make some mistakes
    num_updates = random.randint(0, min(2,len(required_fields)))
    requirements = random.sample(required_fields, num_updates)
    context_details_old = copy.deepcopy(context_details)
    for requirement in requirements:
        context_details[requirement] = get_settings()[requirement]

    # randomly add additional information for certain questions
    for opt, versions in optional_info.items():
        if opt in required_fields and random.randint(0, 1):
            version = random.choice(versions)
            question = question[:-1] + ' ' + version + '?'

    # maybe say hello
    if random.randint(0, 1) == 1:
        curr_dialogue += [random.choice(['hello', 'hi', 'good morning', 'hi there', 'good evening', 'good afternoon'])]
        curr_dialogue += ['1 hello, what can I help you with today?']

    curr_dialogue += [sub_reqs(cur, question, req_sub_dict, context_details)]

    for req_num, requirement in enumerate(additional_qs.keys()):
        if requirement in required_fields:
            if not re.findall(str(requirement), question):
                bot_question = str(req_num + basic_act_size) + ' ' + additional_qs[requirement]
                curr_dialogue += [sub_reqs(cur, bot_question, req_sub_dict, context_details)]
                curr_dialogue += [sub_reqs(cur, requirement, req_sub_dict, context_details)]

    # answer question
    curr_dialogue = answer_quest(curr_dialogue, query, context_details)

    # correct information
    for requirement in requirements:
        context_details[requirement] = context_details_old[requirement]
        curr_dialogue += ['actually i meant ' + sub_reqs(cur, requirement, req_sub_dict, context_details)]
        curr_dialogue += ['2 ok, is there anything else to update?']
    if num_updates > 0:
        curr_dialogue += ['no']
        # answer question after updates
        curr_dialogue = answer_quest(curr_dialogue, query, context_details)

    # maybe say thank you
    if random.randint(0,1) == 1:
        curr_dialogue += [random.choice(['thank you','thanks','ok thanks'])]
        curr_dialogue += ['4 no problem']

    dialogues.append(curr_dialogue)



# save dialogues to text file, in format user_input<\t>bot_response
with open(output+set+'.txt', 'w') as f:
    for chat in dialogues:
        pair = []
        for line in chat:
            pair.append(line)
            if len(pair) == 2:
                f.write('\t'.join(pair))
                f.write("\n")
                pair = []
        f.write("\n")


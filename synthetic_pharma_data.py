import random, sqlite3, re, json
from novartis_config import questions

with open('dailymed_results_prescription.json', encoding='utf-8') as f:
    drugs = json.load(f)

# dailymed_results_prescription.json

# generate synthetic input data
num_drugs = len(drugs)
database = 'NovartisQA'
# NovartisQA
truncate_to = 700


def get_settings():
    #TODO make generic?
    ran_drug = random.choice(entity_dict['<drug_name>']) #random.randint(1, len(entity_dict['<drug_name>']))
    cur.execute('SELECT id FROM Drugs WHERE name = ?', (ran_drug,))
    data = cur.fetchall()
    id = data[0][0]

    cur.execute('''SELECT strength, units, Delivery.method FROM Dosages 
                JOIN Delivery ON delivery_id = Delivery.id
                WHERE drug_id = ?''', (id,))
    data = cur.fetchall()
    delivery_meth = data[0][2]

    rand_version = random.randint(0, len(data) - 1)
    rand_strength = str(data[rand_version][0])
    units = data[rand_version][1]

    cur.execute('''SELECT symptom FROM Symptoms JOIN Complaint 
                ON id = Complaint.symptom_id WHERE Complaint.drug_id = ?''', (id,))
    symptom_data = cur.fetchall()
    ran_symptom = random.choice(symptom_data)[0]

    cur.execute('''SELECT condition FROM Conditions JOIN Contraindications ON id = Contraindications.condition_id
                WHERE Contraindications.drug_id = ?''', (id,))
    condition_data = cur.fetchall()
    ran_condition = random.choice(condition_data)[0]
    ran_settings = {'<drug_name>': ran_drug, '<age_group>': random.choice(entity_dict['<age_group>']),
                    '<delivery>': delivery_meth, '<condition>': ran_condition, '<symptom>': ran_symptom,
                    '<strength>': rand_strength, '<units>': units}
    return ran_settings


# all the following lists would come from the XML data if we had it
entity_dict = {
    '<drug_name>': [name.split()[0] for name in drugs.keys()],
    '<age_group>': ['elderly', 'adult', 'child', 'baby'],
    '<condition>': ['pregnant', 'IBD', 'TB', 'heart_problems', 'liver_problems', 'infection', 'low_blood_pressure',
                    'glaucoma', 'anxiety', 'tourettes'],
    '<symptom>': ['itchy_skin', 'tiredness', 'nausea', 'vomiting', 'bruising', 'bleeding', 'depression', 'anxiety',
                  'vomiting', 'diarrhea', 'insomnia', 'aggression', 'suicidal', 'hallucinations', 'mood_swings', 'fever',
                  'blurred_vision', 'stomach_ache'],
    '<delivery>': ['oral', 'intravenous', 'subcutaneous'],
    '<strength>': [a*10 for a in range(1,10)],
    '<units>': ['grams','mg','mcg']
}

print(entity_dict)

drug_info = {}
dosage_table = []
severities = ['common', 'mild', 'serious', 'dangerous']
periods = ['daily', 'weekly', '4 hours', '6 hours']
for drug_name in drugs.keys():
    short_name = drug_name.split()[0]
    rand_version = random.choice(entity_dict['<delivery>'])
    drug_info[short_name+'_'+rand_version] = \
        {'indication': drugs[drug_name]['indication_usage_description'][:truncate_to],
         'instructions': drugs[drug_name]['dosage_description'][:truncate_to],
         'ingredients': drugs[drug_name]['ingredients'][:truncate_to],
         'storage': 'storage instructions for drug ' + short_name,
         'symptoms': {a:[random.choice(severities),drugs[drug_name]['adverse_description'][:truncate_to]]
                      for a in random.sample(entity_dict['<symptom>'],random.randint(1,5))},
         'contraindications': {a: drugs[drug_name]['adverse_description'][:truncate_to]
                               for a in random.sample(entity_dict['<condition>'],random.randint(1,5))},
         }


    dose_units = random.choice(entity_dict['<units>'])
    per = random.choice(periods)
    for strength in random.sample(entity_dict['<strength>'], random.randint(1, 4)):
        for age, dose_int in zip(entity_dict['<age_group>'], [4, 3, 2, 1]):
            dosage_table.append([short_name, rand_version, strength, dose_units, age, dose_int, per])

print(drug_info)
print(dosage_table)

# save entities to text file, in format type<\t>value ('period', 'severities' are not context info)
with open('entities.txt', 'w') as f:
    for tag_type, values in entity_dict.items():
        # if tag_type == '<drug_name>':
        #     for drug_num in values:
        #         f.write('\t'.join([tag_type, 'drug' + str(drug_num)]))
        #         f.write("\n")
        # else:
        for value in values:
            f.write('\t'.join([tag_type, str(value)]))
            f.write("\n")


# create SQL file
conn = sqlite3.connect(database+'.sqlite')
cur = conn.cursor()

# Delete existing SQL tables
cur.executescript('''
DROP TABLE IF EXISTS Questions;
DROP TABLE IF EXISTS Drugs;
DROP TABLE IF EXISTS Conditions;
DROP TABLE IF EXISTS Ages;
DROP TABLE IF EXISTS Symptoms;
DROP TABLE IF EXISTS Severities;
DROP TABLE IF EXISTS Delivery;
DROP TABLE IF EXISTS Periodicities;
DROP TABLE IF EXISTS Dosages;
DROP TABLE IF EXISTS Complaint;
DROP TABLE IF EXISTS Contraindications;
DROP TABLE IF EXISTS Interactions;
DROP TABLE IF EXISTS Drug_versions;
''')

# Create SQL tables
cur.executescript('''
CREATE TABLE Questions (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    format1 TEXT,
    format2 TEXT,
    format3 TEXT,
    format4 TEXT,
    query TEXT,
    answer TEXT,
    required_fields TEXT
);

CREATE TABLE Drugs (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    name    TEXT UNIQUE, 
    indication TEXT  
);

CREATE TABLE Conditions (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    condition   TEXT UNIQUE
);

CREATE TABLE Ages (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    age_group   TEXT UNIQUE
);


CREATE TABLE Symptoms (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    symptom   TEXT UNIQUE
);

CREATE TABLE Severities (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    severity   TEXT UNIQUE
);

CREATE TABLE Delivery (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    method   TEXT UNIQUE
);

CREATE TABLE Periodicities (
    id  INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,
    periodicity   TEXT UNIQUE
);

CREATE TABLE Drug_versions (
    drug_id INTEGER,
    delivery_id INTEGER,
    instructions TEXT,
    ingredients TEXT,
    storage TEXT,
    PRIMARY KEY (drug_id, delivery_id)
);

CREATE TABLE Dosages (
    drug_id INTEGER,
    delivery_id INTEGER,
    strength INTEGER,
    units TEXT,
    age_id INTEGER,
    dose INTEGER,
    period_id INTEGER,
    PRIMARY KEY (drug_id, delivery_id, strength, age_id)
);

CREATE TABLE Complaint (
    symptom_id INTEGER,
    drug_id INTEGER,
    severity_id INTEGER,
    advice TEXT,
    PRIMARY KEY (drug_id, symptom_id)
);

CREATE TABLE Contraindications (
    drug_id INTEGER,
    condition_id INTEGER,
    advice TEXT,
    PRIMARY KEY (drug_id, condition_id)
);

CREATE TABLE Interactions (
    drug_id INTEGER,
    other_drug_id INTEGER,
    advice TEXT,
    PRIMARY KEY (drug_id, other_drug_id)
);
''')

# populate SQL tables
# question table
for question in questions.values():
    [format1, format2, format3, format4, query, answer, required_fields] = question
    cur.execute('''INSERT OR IGNORE INTO Questions (format1, format2, format3, format4, query, answer, required_fields) 
        VALUES ( ?, ?, ?, ?, ?, ?, ? )''', (format1, format2, format3, format4, query, answer, required_fields))

# all data tables except dosages
for drug_name, info_dict in drug_info.items():
    # extract name, delivery method and add to database if new, get IDs from database
    [name, delivery_meth] = re.split('_', drug_name)
    cur.execute('''INSERT OR IGNORE INTO Drugs (name, indication)
        VALUES ( ?, ? )''', (name, info_dict['indication']))
    cur.execute('''INSERT OR IGNORE INTO Delivery (method)
        VALUES ( ? )''', (delivery_meth,))
    conn.commit()
    cur.execute('SELECT id FROM Drugs WHERE name = ?', (name,))
    data = cur.fetchall()
    drug_id = data[0][0]
    cur.execute('SELECT id FROM Delivery WHERE method = ?', (delivery_meth,))
    data = cur.fetchall()
    delivery_id = data[0][0]

    # create unique info for drug version
    cur.execute('''INSERT OR IGNORE INTO Drug_versions (drug_id, delivery_id, instructions, ingredients, storage) 
        VALUES ( ?, ?, ?, ?, ? )''', (drug_id, delivery_id, info_dict['instructions'], info_dict['ingredients'],
                                      info_dict['storage']))

    # process side-effects
    for symptom, [severity, advice] in info_dict['symptoms'].items():
        # update symptom table
        cur.execute('''INSERT OR IGNORE INTO Symptoms (symptom)
            VALUES ( ? )''', (symptom,))
        cur.execute('SELECT id FROM Symptoms WHERE symptom = ?', (symptom,))
        conn.commit()
        data = cur.fetchall()
        symptom_id = data[0][0]

        # update severity table
        cur.execute('''INSERT OR IGNORE INTO Severities (severity)
            VALUES ( ? )''', (severity,))
        conn.commit()
        cur.execute('SELECT id FROM Severities WHERE severity = ?', (severity,))
        data = cur.fetchall()
        severity_id = data[0][0]

        #update complaint table
        cur.execute('''INSERT OR IGNORE INTO Complaint (drug_id, symptom_id, severity_id, advice) 
            VALUES ( ?, ?, ?, ? )''', (drug_id, symptom_id, severity_id, advice))

    # process contraindications
    for condition, advice in info_dict['contraindications'].items():
        # update condition table
        cur.execute('''INSERT OR IGNORE INTO Conditions (condition)
            VALUES ( ? )''', (condition,))
        conn.commit()
        cur.execute('SELECT id FROM Conditions WHERE condition = ?', (condition,))
        data = cur.fetchall()
        condition_id = data[0][0]

        #update contraindication table
        cur.execute('''INSERT OR IGNORE INTO Contraindications (drug_id, condition_id, advice) 
            VALUES ( ?, ?, ? )''', (drug_id, condition_id, advice))

# dosage table
for drug_name, delivery, strength, dose_units, age, dose_int, per in dosage_table:
    cur.execute('SELECT id FROM Drugs WHERE name = ?', (drug_name,))
    conn.commit()
    data = cur.fetchall()
    drug_id = data[0][0]

    # update ages table
    cur.execute('''INSERT OR IGNORE INTO Ages (age_group)
        VALUES ( ? )''', (age,))
    cur.execute('SELECT id FROM Ages WHERE age_group = ?', (age,))
    conn.commit()
    data = cur.fetchall()
    age_id = data[0][0]

    # update periodicities table
    cur.execute('''INSERT OR IGNORE INTO Periodicities (periodicity)
        VALUES ( ? )''', (per,))
    conn.commit()
    cur.execute('SELECT id FROM Periodicities WHERE periodicity = ?', (per,))
    data = cur.fetchall()
    period_id = data[0][0]

    #get delivery_id
    cur.execute('SELECT id FROM Delivery WHERE method = ?', (delivery,))
    data = cur.fetchall()
    delivery_id = data[0][0]


    # update dosages table
    cur.execute('''INSERT OR IGNORE INTO Dosages (drug_id, delivery_id, strength, units, age_id, dose, period_id) 
        VALUES ( ?, ?, ?, ?, ?, ?, ? )''', (drug_id, delivery_id, strength, dose_units, age_id, dose_int, period_id))

conn.commit()



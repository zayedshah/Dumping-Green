# configuration templates for dialog_gen.py, this file does not return anything

# question templates [format1, format2, format3, format4, query, fetch command, required entities]
# 'answer_text' will be returned by the following commands: cur.execute(query); exec(fetch command);
questions = {}
questions['question1'] = ['What is <drug_name>',
                         '',
                         '',
                         '',
                         'SELECT indication FROM Drugs WHERE name = \'<drug_name>\'',
                         'answer_text = cur.fetchall()[0][0]',
                         '<drug_name>'
                         ]
questions['question2'] = ['Should I take <drug_name>',
                         'Is <drug_name> safe',
                         'can i take <drug_name>',
                         '',
                         '''SELECT advice 
                         FROM Contraindications JOIN Drugs JOIN Conditions 
                         ON drug_id = Drugs.id AND condition_id = Conditions.id
                         WHERE Drugs.name = \'<drug_name>\' AND Conditions.condition = \'<condition>\'''',
                         'answer_text = cur.fetchall()[0][0]',
                         '<drug_name>,<condition>'
                         ]
# questions['question3'] = ['can i use <drug_name> along with <other_drug_name>',
#                          'is it safe to take <drug_name> and <other_drug_name> together',
#                          'do <drug_name> and <other_drug_name> interact',
#                          'does <other_drug_name> stop <drug_name> from working',
#                          '''SELECT Interactions.advice
#                          FROM Interactions
#                          JOIN Drugs d1 ON Interactions.drug_id = d1.id
#                          JOIN Drugs d2 ON Interactions.other_drug_id = d2.id
#                          WHERE d1.name = ? AND d2.name = ?, (<drug_name>, <other_drug_name>)''',
#                          'print(cur.fetchall[0]),'
#                          '<drug_name>,<other_drug_name>'
#                          ]
questions['question4'] = ['how do I use <drug_name>',
                         'what do I do with <drug_name>',
                         'how do I take <drug_name>',
                         '',
                         '''SELECT instructions 
                         FROM Drug_versions JOIN Drugs JOIN Delivery 
                         ON drug_id = Drugs.id AND delivery_id = Delivery.id
                         WHERE Drugs.name = \'<drug_name>\' AND Delivery.method = \'<delivery>\' ''',
                         'answer_text = cur.fetchall()[0][0]',
                         '<drug_name>,<delivery>'
                         ]
questions['question5'] = ['how many <drug_name> is the correct dose',
                         'what is the dose for <drug_name>',
                         'how much <drug_name> should i take',
                         '',
                         '''SELECT Drugs.name, dose, units, Periodicities.periodicity
                         FROM Dosages JOIN Drugs JOIN Conditions JOIN Periodicities JOIN Delivery JOIN Ages
                         ON Drugs.id = drug_id AND Periodicities.id = period_id AND age_id = Ages.id
                         WHERE Drugs.name = \'<drug_name>\' AND Ages.age_group = \'<age_group>\' AND Delivery.method = \'<delivery>\' 
                         AND strength = \'<strength>\' ''',
                         'data = cur.fetchall(); answer_text = \'The dosage for %s is %s %s per %s\' % data[0]',
                         '<drug_name>,<age_group>,<delivery>,<strength>'
                         ]
questions['question6'] = ['what are the side effects of <drug_name>',
                         'is it normal to have <symptom> with <drug_name>',
                         'does <drug_name> cause <symptom>',
                         'is <symptom> a common side effect of <drug_name>',
                         '''SELECT Symptoms.symptom, Severities.severity, Drugs.name, advice
                         FROM Complaint JOIN Drugs JOIN Symptoms JOIN Severities 
                         ON Drugs.id = drug_id AND Symptoms.id = symptom_id AND Severities.id = severity_id
                         WHERE Drugs.name = \'<drug_name>\' AND Symptoms.symptom = \'<symptom>\' ''',
                         'data = cur.fetchall(); answer_text = \'%s is a %s side effect of %s. %s\' % data[0]',
                         '<drug_name>,<symptom>'
                         ]
questions['question7'] = ['how do I store <drug_name>',
                         'where should I keep <drug_name>',
                         'do I need to keep <drug_name> in the fridge',
                         'do I need to keep <drug_name> in the dark',
                         '''SELECT storage
                         FROM Drug_versions JOIN Drugs JOIN Delivery  
                         ON Drugs.id = drug_id AND Delivery.id = delivery_id 
                         WHERE Drugs.name = \'<drug_name>\' AND Delivery.method = \'<delivery>\'''',
                         'answer_text = cur.fetchall()[0][0]',
                         '<drug_name>,<delivery>'
                         ]
questions['question8'] = ['what are the ingredients of <drug_name>',
                         'what is in <drug_name>',
                         'does <drug_name> contain <ingredient>',
                         '',
                         '''SELECT ingredients
                         FROM Drug_versions JOIN Drugs JOIN Delivery  
                         ON Drugs.id = drug_id AND Delivery.id = delivery_id 
                         WHERE Drugs.name = \'<drug_name>\' AND Delivery.method = \'<delivery>\'''',
                         'answer_text = cur.fetchall()[0][0]',
                         '<drug_name>,<delivery>'
                         ]

# additional info which the user may provide
optional_info = {'<condition>': [
                                ' if I have <condition>',
                                ' if I am <condition>',
                                ' for someone with <condition>'
                                 ],
                 '<age_group>': ['for a <age_group>']
                 }


# for bot to ask for required entities (keys to match those given in questions dictionary)
additional_qs = {'<age_group>': 'for which age group?',
                 '<condition>': '<drug_name> may not be recommended for people with <contraindications>, do you have any of these?',
                 '<delivery>': 'what version of <drug_name> is that, for example: oral, intravenous or subcutaneous?',
                 '<strength>': 'what is the strength of the drug?',
                 '<symptom>': 'these are some common side effects of <drug_name>: <side_effects>, are you seeing any other symptoms?'
                 }

# dictionary to call information from SQL in order to sub into bot responses, {<entity>: [query, fetch command],...}
req_sub_dict = {
    '<drug_name>': ['result = context_details[\'<drug_name>\']', 'sub_data = result'],
    '<symptom>': ['result = context_details[\'<symptom>\']', 'sub_data = result'],
    '<age_group>': ['result = context_details[\'<age_group>\']', 'sub_data = result'],
    '<condition>': ['result = context_details[\'<condition>\']', 'sub_data = result'],
    '<delivery>': ['result = context_details[\'<delivery>\']', 'sub_data = result'],
    '<strength>': ['''result = str(context_details['<strength>'])''', 'sub_data = result'],
    '<units>': ['''result = str(context_details['<units>'])''', 'sub_data = result'],
    '<side_effects>': ['''cur.execute(\'\'\'SELECT Symptoms.symptom FROM Complaint JOIN Symptoms JOIN Severities JOIN Drugs
                                  ON symptom_id = Symptoms.id AND severity_id = Severities.id AND drug_id = Drugs.id
                                  WHERE Drugs.name = ? AND Severities.severity = \'common\' \'\'\', 
                                  (context_details['<drug_name>'],))''',
                       '''result = cur.fetchall(); sub_data = ', '.join([a for (a,) in result])'''],
    '<contraindications>': ['''cur.execute(\'\'\'SELECT  Conditions.condition FROM Contraindications JOIN Conditions JOIN Drugs
                                                  ON condition_id = Conditions.id AND drug_id = Drugs.id
                                                  WHERE Drugs.name = ? \'\'\', (context_details['<drug_name>'],))''',
                            '''result = cur.fetchall(); sub_data = ', '.join([a for (a,) in result])''']
}

import csv

with open('Training.csv', 'r') as f:
    d_reader = csv.DictReader(f)

    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames


f = open('list_symptoms.txt', 'w')
f.write(str(headers))

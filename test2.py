import csv

with open('data/Handpositions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['position_id', 'description'])

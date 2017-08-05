import csv
import sys

if len(sys.argv) == 3:
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    output_file = open(output_file_name, 'w')

    with open(input_file_name) as input_file:
        reader = csv.reader(input_file)
        for row in reader:
            if 'IMG' not in row[0]:
                continue

            path = 'IMG/'

            steering_center = float(row[3])
            for i in range(0, 3):
                output_file.write(path + row[i].split('/')[-1] + ',')

            for i in range(3, 7):
                output_file.write(row[i] + ',')
            output_file.write('\n')
    output_file.close()
else:
    print('usage: python clean_csvfile.py ./training_data/track2_set0/driving_log.csv ./training_data/track2_set0/driving_log_new.csv')
import argparse
import csv

argparser = argparse.ArgumentParser()
argparser.add_argument("--csv", type=str, default="./con.csv")
args = argparser.parse_args()

# Initialize counters for each hair and eye color
hair_counts = {
    "orange": 0,
    "white": 0,
    "aqua": 0,
    "gray": 0,
    "green": 0,
    "red": 0,
    "purple": 0,
    "pink": 0,
    "blue": 0,
    "black": 0,
    "brown": 0,
    "blonde": 0,
}
eye_counts = {
    "gray": 0,
    "black": 0,
    "orange": 0,
    "pink": 0,
    "yellow": 0,
    "aqua": 0,
    "purple": 0,
    "green": 0,
    "brown": 0,
    "red": 0,
    "blue": 0,
}

# Open the CSV file
with open(args.csv, newline="") as csvfile:
    reader = csv.reader(csvfile)

    for row in reader:
        if reader.line_num == 1:
            continue

        hair_counts["orange"] += int(row[1])
        hair_counts["white"] += int(row[2])
        hair_counts["aqua"] += int(row[3])
        hair_counts["gray"] += int(row[4])
        hair_counts["green"] += int(row[5])
        hair_counts["red"] += int(row[6])
        hair_counts["purple"] += int(row[7])
        hair_counts["pink"] += int(row[8])
        hair_counts["blue"] += int(row[9])
        hair_counts["black"] += int(row[10])
        hair_counts["brown"] += int(row[11])
        hair_counts["blonde"] += int(row[12])
        eye_counts["gray"] += int(row[13])
        eye_counts["black"] += int(row[14])
        eye_counts["orange"] += int(row[15])
        eye_counts["pink"] += int(row[16])
        eye_counts["yellow"] += int(row[17])
        eye_counts["aqua"] += int(row[18])
        eye_counts["purple"] += int(row[19])
        eye_counts["green"] += int(row[20])
        eye_counts["brown"] += int(row[21])
        eye_counts["red"] += int(row[22])
        eye_counts["blue"] += int(row[23])

print("Hair color counts:")
for color, count in hair_counts.items():
    print(f"{color}: {count}")

print("Eye color counts:")
for color, count in eye_counts.items():
    print(f"{color}: {count}")

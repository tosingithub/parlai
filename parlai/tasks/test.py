import csv

def pl():
    prompt = ""                 # initialize prompt message
    cnter = 0                   # initialize conversation counter
    new_episode = False
    line_skipper = False        # to skip 1st row of curated dailog data with header: line
    with open("data/yoruba_dialog.csv", "r+") as data_file:
        #yodialog = csv.reader(data_file, delimiter=',')
        for row in data_file.readlines():
            row_array = row.rstrip('\n').split(';')
            #first_item = row_array[1]
            if cnter == 0:
                prompt = row_array[1]
                cnter += 1
                if str.lower(row_array[0]) == 'true':
                    new_episode = True
                    print("Here true")
                else:
                    new_episode = False
                    print("Here false")
                print("1st",new_episode)
            else:
                # new_episode == True if str.lower(row_array[0]) == 'true' else False
                print("2nd",new_episode)
                yield {"text": prompt, "labels": row_array[1]}, new_episode
                cnter = 0


if __name__ == "__main__":
    for i in pl():
        print(i)
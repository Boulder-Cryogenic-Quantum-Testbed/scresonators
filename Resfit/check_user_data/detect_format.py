import numpy as np

#set file location
dir = "Your path to file here" #path to directory with data, make sure to use /
filename = "Your file name here"
filepath = dir+"/"+filename

#read in file
file = open(filepath, 'r')
line = file.readline()
while len(line) < 2:
    line = file.readline()

#create a new line from the first line of code that contains no numbers to test for header/delimiters
newline = ""
for letter in line:
    if (ord(letter)<45 or ord(letter)>57) and ord(letter) != 10 and ord(letter) != 101 and ord(letter) != 69:
        newline = newline + letter
    if ord(letter)==47:
        newline = newline + letter

#test if there is a header in user data, test if user has correct number of columns and test if the user is delimiting data correctly
correct_format = False
delimiter = ","
if len(newline) == 2:
    if newline[0] == newline[1]: #if delimited by something else besides commas
        if newline[0] == ",":
            correct_format = True
        else:
            print("File is delimited by: '" + newline[0] + "'")
            print("Would you like to change the delimeter to a comma? Type y or n:")
            user_input = input().lower()
            if user_input == "y" or user_input == "yes":
                delimiter = newline[0]
                correct_format = True
                print("Changed delimiter to comma")
            else:
                print("File format fix failed. Please follow correct format by delimiting data with a comma.")
    else:
        #if header is detected
        print("Header detected. Please follow the correct format and remove any headers.")
elif len(newline) > 2:
    if len(line)>len(newline):
        all_equal = True
        prev_letter = newline[0]
        for letter in newline:
            if letter != prev_letter: #if more than 3 columnds detected
                all_equal = False
            prev_letter = letter
        if all_equal:
            print("More than 3 columns detected. Please have only three columns following standard data format.")
        else:
            print("Header detected. Please follow the correct format and remove any headers.")
else:
    print("Header detected. Please follow the correct format and remove any headers.")


file.close()

if not correct_format:
    quit()

#detect if numbers in columns follow specified format
print("Detecting if column numbers are of correct format:")

changes_made = False

data = np.genfromtxt(filepath, delimiter = delimiter)
frequency = data.T[0]
magnitude = data.T[1]
phase = data.T[2]

if len(frequency)<2:
    print("Only one line found for data. Please include more than one line.")

for f in frequency:
    if f > 100000000:
        print("Frequency likely not in GHz.")
        print("Would you like to convert to GHz from Hz? Type y or n:")
        user_input = input().lower()
        if user_input == "y" or user_input == "yes":
            frequency = np.multiply(frequency,1/1000000000)
            print("Converted frequency from Hz to GHz")
            changes_made = True
        else:
            print("Frequency not converted")

        break

for w in phase:
    if w > np.pi*2 or w < -np.pi*2:
        print("Phase detected to be in degrees.")
        print("Would you like to convert to radians from degrees? Type y or n:")
        user_input = input().lower()
        if user_input == "y" or user_input == "yes":
            phase = np.multiply(phase,np.pi/180)
            print("Converted phase to radians from degrees")
            changes_made = True
        else:
            print("Phase not converted")
        break

if changes_made == True:
    #write changes to new file
    file = open(filename + "_edited.csv","w")
    count = 0
    for f in frequency:
        file.write(str(f)+','+str(magnitude[count])+','+str(phase[count])+'\n')
        count = count + 1
    print("Output corrected file to " + filename + "_edited.csv")
elif (delimiter == ","):
    print("No changes made")

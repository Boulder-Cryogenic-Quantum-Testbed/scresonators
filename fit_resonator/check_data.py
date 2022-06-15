import numpy as np
import os


def file(filename):
    """Check data from a file

    Args:
        filename (str): Relative path to your data file

    Returns:
        None
    """


    # read in file
    file = open(filename, 'r')

    line = file.readline()
    while len(line) < 2:
        line = file.readline()

    # create a new line from the first line of code that contains no numbers to test for header/delimiters
    newline = ""
    for letter in line:
        if (ord(letter) < 45 or ord(letter) > 57) and ord(letter) != 10 and ord(letter) != 101 and ord(letter) != 69:
            newline = newline + letter
        if ord(letter) == 47:
            newline = newline + letter

    # test if there is a header in user data, test if user has correct number of columns and test if the user is delimiting data correctly
    correct_format = False
    delimiter = ","
    if len(newline) == 2:
        if newline[0] == newline[1]:  # if delimited by something else besides commas
            if newline[0] == ",":
                correct_format = True
            else:
                print("File is delimited by: '" + newline[0] + "'")
                print("Would you like to change the delimiter to a comma? Type y or n:")
                user_input = input().lower()
                if user_input == "y" or user_input == "yes":
                    delimiter = newline[0]
                    correct_format = True
                    print("Changed delimiter to comma")
                else:
                    print("File format fix failed. Please follow correct format by delimiting data with a comma.")
        else:
            # if header is detected
            print("Header detected. Please follow the correct format and remove any headers.")
    elif len(newline) > 2:
        if len(line) > len(newline):
            all_equal = True
            prev_letter = newline[0]
            for letter in newline:
                if letter != prev_letter:  # if more than 3 columns detected
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

    # detect if numbers in columns follow specified format
    print("Detecting if column numbers are of correct format:")

    data = np.genfromtxt(filename, delimiter=delimiter)
    frequency = data.T[0]
    magnitude = data.T[1]
    phase = data.T[2]

    parse(frequency, magnitude, phase, filename, True)


def raw(freq, mag, phase):
    """Check raw data

    Args:
        freq (array or array-like): Monotonically increasing frequency (GHz)
        mag (array or array-like): Magnitude of S21 (dB) log mag
        phase (array or array-like): Phase of S21 (radians)

    Returns:
        None if no changes made
        (freq, mag, phase) tuple if changes from raw are made
    """

    parse(freq, mag, phase)


def parse(frequency, magnitude, phase, filepath=None, file_flag=False):
    changes_made = False
    if len(frequency) < 2:
        print("Only one line found for data. Please include more than one line.")

    for f in frequency:
        if f > 100000000:
            print("Frequency likely not in GHz.")
            print("Would you like to convert to GHz from Hz? Type y or n:")
            user_input = input().lower()
            if user_input == "y" or user_input == "yes":
                frequency = np.multiply(frequency, 1 / 1000000000)
                print("Converted frequency from Hz to GHz")
                changes_made = True
            else:
                print("Frequency not converted")

            break

    for w in phase:
        if w > np.pi * 2 or w < -np.pi * 2:
            print("Phase detected to be in degrees.")
            print("Would you like to convert to radians from degrees? Type y or n:")
            user_input = input().lower()
            if user_input == "y" or user_input == "yes":
                phase = np.multiply(phase, np.pi / 180)
                print("Converted phase to radians from degrees")
                changes_made = True
            else:
                print("Phase not converted")
            break

    if changes_made:
        if file_flag:
            # write changes to new file
            filename, extension = os.path.splitext(filepath)
            file = open(filename + "_edited" + extension, "w")
            count = 0
            for f in frequency:
                file.write(str(f) + ',' + str(magnitude[count]) + ',' + str(phase[count]) + '\n')
                count = count + 1
            print("Output corrected file to " + filename + "_edited" + extension)
            file.close()
        else:
            # Return changes as output
            return frequency, magnitude, phase

##Detect Format:

A simple script to detect if the user's data file is of the correct format.

To use, simply change the path variables dir to directory with file, and filename to the name of the file to be checked.

This code will check for:

* Header (not currently part of standard format)
* Correct number of columns
* Correct delimiter (currently set to ',')
* Containing more than 1 line of data
* Frequency in GHz (determined by checking if frequency is above 10^8 in magnitude)
* Phase in radians (determined by checking if phase less/greater than 2pi)

>Code does not check if magnitude is in dB

The code will prompt the user to check if they would like to make a new file with an edited version of their data using the correct format for each individual change.
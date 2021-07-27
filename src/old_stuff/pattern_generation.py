"""
Generate character pattern and save as text file
"""

#pattern = 5 * (100 * "a " + "# ")
pattern = 5 * (100 * ("abc  " + (3 * "o o o ") + "xyz ") + "# ")

with open("data/nested_abcxyz.txt", "w") as f:
    f.write(pattern)
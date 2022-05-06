# sql1.py
"""Volume 1: SQL 1 (Introduction).
Bryant McArthur
Sec 002
March 15
"""

import sqlite3 as sql
import csv
from matplotlib import pyplot as plt
import numpy as np

# Problems 1, 2, and 4
def student_db(db_file="students.db", student_info="student_info.csv",
                                      student_grades="student_grades.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the tables MajorInfo, CourseInfo, StudentInfo, and StudentGrades from
    the database (if they exist). Recreate the following (empty) tables in the
    database with the specified columns.

        - MajorInfo: MajorID (integers) and MajorName (strings).
        - CourseInfo: CourseID (integers) and CourseName (strings).
        - StudentInfo: StudentID (integers), StudentName (strings), and
            MajorID (integers).
        - StudentGrades: StudentID (integers), CourseID (integers), and
            Grade (strings).

    Next, populate the new tables with the following data and the data in
    the specified 'student_info' 'student_grades' files.

                MajorInfo                         CourseInfo
            MajorID | MajorName               CourseID | CourseName
            -------------------               ---------------------
                1   | Math                        1    | Calculus
                2   | Science                     2    | English
                3   | Writing                     3    | Pottery
                4   | Art                         4    | History

    Finally, in the StudentInfo table, replace values of −1 in the MajorID
    column with NULL values.

    Parameters:
        db_file (str): The name of the database file.
        student_info (str): The name of a csv file containing data for the
            StudentInfo table.
        student_grades (str): The name of a csv file containing data for the
            StudentGrades table.
    """
    #Read CSV files and get row info
    with open("student_info.csv", 'r') as infile:
        inforows = list(csv.reader(infile))

    with open("student_grades.csv", 'r') as infile:
        graderows = list(csv.reader(infile))
        
    #Hardcode other row data
    majorrows = [(1, "Math"), (2, "Science"), (3, "Writing"), (4, "Art")]
    courserows = [(1, "Calculus"), (2, "English"), (3, "Pottery"), (4, "History")]
    
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            
            #Drop Tables
            cur.execute("DROP TABLE IF EXISTS MajorInfo;")
            cur.execute("DROP TABLE IF EXISTS CourseInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentInfo;")
            cur.execute("DROP TABLE IF EXISTS StudentGrades;")
            
            #Create Tables
            cur.execute("CREATE TABLE MajorInfo (MajorID INTEGER, MajorName TEXT);")
            cur.execute("CREATE TABLE CourseInfo (CourseID INTEGER, CourseName TEXT);")
            cur.execute("CREATE TABLE StudentInfo (StudentID INTEGER, StudentName TEXT, MajorID INTEGER);")
            cur.execute("CREATE TABLE StudentGrades (StudentID INTEGER, CourseID INTEGER, Grade TEXT);")
            
            #Populate Tables
            cur.executemany("INSERT INTO MajorInfo VALUES(?,?);", majorrows)
            cur.executemany("INSERT INTO CourseInfo VALUES(?,?);", courserows)
            cur.executemany("INSERT INTO StudentInfo Values(?,?,?);", inforows)
            cur.executemany("INSERT INTO StudentGrades Values(?,?,?);", graderows)
            
            #Set Null values
            cur.execute("UPDATE StudentInfo SET MajorID = NULL WHERE MajorID == -1")
            
        conn.commit()
            
    finally:
        conn.close()

#student_db()

# Problems 3 and 4
def earthquakes_db(db_file="earthquakes.db", data_file="us_earthquakes.csv"):
    """Connect to the database db_file (or create it if it doesn’t exist).
    Drop the USEarthquakes table if it already exists, then create a new
    USEarthquakes table with schema
    (Year, Month, Day, Hour, Minute, Second, Latitude, Longitude, Magnitude).
    Populate the table with the data from 'data_file'.

    For the Minute, Hour, Second, and Day columns in the USEarthquakes table,
    change all zero values to NULL. These are values where the data originally
    was not provided.

    Parameters:
        db_file (str): The name of the database file.
        data_file (str): The name of a csv file containing data for the
            USEarthquakes table.
    """
    #Read in the earthquake data
    with open(data_file, 'r') as infile:
        rows = list(csv.reader(infile))
        
        
    try:
        with sql.connect(db_file) as conn:
            cur = conn.cursor()
            
            #Drop Tables
            cur.execute("DROP TABLE IF EXISTS USEarthquakes;")
            
            #Create Tables
            cur.execute("CREATE TABLE USEarthquakes (Year INTEGER, Month INTEGER, \
                        Day INTEGER, Hour INTEGER, Minute INTEGER, Second INTEGER, \
                            Latitude REAL, Longitude REAL, Magnitude REAL);")
        
            #Populate Tables
            cur.executemany("INSERT INTO USEarthquakes VALUES(?,?,?,?,?,?,?,?,?);", rows)
            
            #Delete zero magnitude rows
            cur.execute("DELETE FROM USEarthquakes WHERE Magnitude==0")
            
            #Set Null values
            cur.execute("UPDATE USEarthquakes SET Day = NULL WHERE Day == 0")
            cur.execute("UPDATE USEarthquakes SET Hour = NULL WHERE Hour == 0")
            cur.execute("UPDATE USEarthquakes SET Minute = NULL WHERE Minute == 0")
            cur.execute("UPDATE USEarthquakes SET Second = NULL WHERE Second == 0")
            
            conn.commit()
            
    finally:
        conn.close()
        
#earthquakes_db()

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for all tuples of the form (StudentName, CourseName)
    where that student has an 'A' or 'A+'' grade in that course. Return the
    list of tuples.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #Connect to database and set cursor
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #Execute this confusing code to get what we want
    cur.execute("SELECT SI.StudentName,CI.CourseName FROM StudentInfo AS SI, \
                CourseInfo AS CI, StudentGrades AS SG WHERE SI.StudentID == SG.StudentID \
                AND CI.CourseID == SG.CourseID AND Grade IN ('A', 'A+');")
    
    #Fetch and return
    A = cur.fetchall()
    conn.close()
    return A

#print(prob5())


# Problem 6
def prob6(db_file="earthquakes.db"):
    """Create a single figure with two subplots: a histogram of the magnitudes
    of the earthquakes from 1800-1900, and a histogram of the magnitudes of the
    earthquakes from 1900-2000. Also calculate and return the average magnitude
    of all of the earthquakes in the database.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (float): The average magnitude of all earthquakes in the database.
    """
    #Connect to database and set cursor
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    #Get 1800 Magnitudes
    cur.execute("SELECT E.Magnitude FROM USEarthquakes AS E WHERE Year < 1900 AND Year >= 1800")
                
    #Fetch and Unravel
    mag1800s = cur.fetchall()
    mag1800s = np.ravel(mag1800s)
    
    #Get 1900 Magnitudes
    cur.execute("SELECT E.Magnitude FROM USEarthquakes AS E WHERE Year < 2000 AND Year >= 1900")
        
    #Fetch and Unravel
    mag1900s = cur.fetchall()
    mag1900s = np.ravel(mag1900s)
    
    #Get the Average Magnitude
    cur.execute("SELECT AVG(E.Magnitude) FROM USEarthquakes AS E")
    
    #PLOT IT
    plt.subplot(121)
    plt.hist(mag1800s)
    plt.subplot(122)
    plt.hist(mag1900s)
    plt.suptitle("Earthquakes in 1800s and 1900s")
    
    plt.show()
    
    return cur.fetchall()[0][0]
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

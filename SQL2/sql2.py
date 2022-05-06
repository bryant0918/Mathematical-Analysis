# solutions.py
"""Volume 1: SQL 2.
Bryant McArthur
Sec 002
March 29, 2022
"""

import sqlite3 as sql
import numpy as np

# Problem 1
def prob1(db_file="students.db"):
    """Query the database for the list of the names of students who have a
    'B' grade in any course. Return the list.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a student name.
    """
    #Connect to cursor and database
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    cur.execute("SELECT SI.StudentName "
                "FROM StudentInfo AS SI INNER JOIN StudentGrades AS SG "
                "ON SI.StudentID == SG.StudentID "
                "WHERE SG.Grade == 'B';")
    
    Bstudents = cur.fetchall()
    
    return [Bstudents[i][0] for i in range(len(Bstudents))]

#print(prob1())

# Problem 2
def prob2(db_file="students.db"):
    """Query the database for all tuples of the form (Name, MajorName, Grade)
    where 'Name' is a student's name and 'Grade' is their grade in Calculus.
    Only include results for students that are actually taking Calculus, but
    be careful not to exclude students who haven't declared a major.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #Connect to cursor and database
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    cur.execute("SELECT SI.StudentName, MI.MajorName, SG.Grade "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                "ON SI.MajorID == MI.MajorID "
                "INNER JOIN StudentGrades AS SG "
                "ON SI.StudentID == SG.StudentID "
                "WHERE SG.CourseID == 1;")
    
    calc = cur.fetchall()
    
    return calc

#print(prob2())

# Problem 3
def prob3(db_file="students.db"):
    """Query the database for the list of the names of courses that have at
    least 5 students enrolled in them.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): a list of strings, each of which is a course name.
    """
    #Connect to cursor and database
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    cur.execute("SELECT CI.CourseName "
                "FROM CourseInfo AS CI INNER JOIN StudentGrades AS SG "
                "ON CI.CourseID == SG.CourseID "
                "GROUP BY CI.CourseName "
                "HAVING COUNT(*) >= 5;")
   
    courses = cur.fetchall()
    
    return [courses[i][0] for i in range(len(courses))]

#print(prob3())


# Problem 4
def prob4(db_file="students.db"):
    """Query the given database for tuples of the form (MajorName, N) where N
    is the number of students in the specified major. Sort the results in
    descending order by the counts N, then in alphabetic order by MajorName.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #Connect to cursor and database
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    cur.execute("SELECT MI.MajorName, COUNT(*) AS num_students "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                "ON MI.MajorID == SI.MajorID "
                "GROUP BY MI.MajorID "
                "ORDER BY num_students DESC, MI.MajorName ASC;")
    
    result = cur.fetchall()
    
    return result

#print(prob4())

# Problem 5
def prob5(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, MajorName) where
    the last name of the specified student begins with the letter C.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #Connect to cursor and database
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    cur.execute("SELECT SI.StudentName, MI.MajorName "
                "FROM StudentInfo AS SI LEFT OUTER JOIN MajorInfo AS MI "
                "ON MI.MajorID == SI.MajorID "
                "WHERE SI.StudentName LIKE '% C%';")
    
    Cstudents = cur.fetchall()
    
    return Cstudents

#print(prob5())


# Problem 6
def prob6(db_file="students.db"):
    """Query the database for tuples of the form (StudentName, N, GPA) where N
    is the number of courses that the specified student is in and 'GPA' is the
    grade point average of the specified student according to the following
    point system.

        A+, A  = 4.0    B  = 3.0    C  = 2.0    D  = 1.0
            A- = 3.7    B- = 2.7    C- = 1.7    D- = 0.7
            B+ = 3.4    C+ = 2.4    D+ = 1.4

    Order the results from greatest GPA to least.

    Parameters:
        db_file (str): the name of the database to connect to.

    Returns:
        (list): the complete result set for the query.
    """
    #Connect to cursor and database
    conn = sql.connect(db_file)
    cur = conn.cursor()
    
    cur.execute("SELECT SI.StudentName, COUNT(*) AS num_courses, AVG( "
                "CASE SG.Grade "
                    "WHEN 'A+' THEN 4.0 "
                    "WHEN 'A' THEN 4.0 "
                    "WHEN 'A-' THEN 3.7 "
                    "WHEN 'B+' THEN 3.4 "
                    "WHEN 'B' THEN 3.0 "
                    "WHEN 'B-' THEN 2.7 "
                    "WHEN 'C+' THEN 2.4 "
                    "WHEN 'C' THEN 2.0 "
                    "WHEN 'C-' THEN 1.7 "
                    "WHEN 'D+' THEN 1.4 "
                    "WHEN 'D' THEN 1.0 "
                    "WHEN 'D-' THEN 0.7 "
                    "END) AS GPA "
                "FROM StudentGrades AS SG "
                "INNER JOIN StudentInfo AS SI "
                "ON SI.StudentID == SG.StudentID "
                "GROUP BY SG.StudentID "
                "ORDER BY GPA DESC;")
    
    results = cur.fetchall()
    
    return results

#print(prob6())
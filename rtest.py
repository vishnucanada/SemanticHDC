# Assignments scores and max scores
assignment_scores = {
    "Assignment1": 52.50,  # out of 65
    "Assignment2": 20.00,  # out of 35
    "Assignment3": 17.50,  # out of 30
    "Assignment4": 47.00,  # out of 60
    "Assignment5": 23.50,  # out of 35
    # Assignment 6 has no score, will exclude it
}

assignment_max_scores = {
    "Assignment1": 65,
    "Assignment2": 35,
    "Assignment3": 30,
    "Assignment4": 60,
    "Assignment5": 35
}

# Quizzes scores and max scores
quiz_scores = {
    "Quiz1": 3.00,
    "Quiz2": 2.00,
    "Quiz3": 4.00,
    "Quiz4": 5.00,
    "Quiz5": 5.00,
    "Quiz6": 3.00,
    "Quiz7": 3.00,
    "Quiz8": 2.83,
    "Quiz9": 2.00,
    "Quiz10": 5.00
}

quiz_max_scores = {
    "Quiz1": 5,
    "Quiz2": 5,
    "Quiz3": 5,
    "Quiz4": 5,
    "Quiz5": 5,
    "Quiz6": 5,
    "Quiz7": 5,
    "Quiz8": 5,
    "Quiz9": 5,
    "Quiz10": 5
}

# Class averages for quizzes (from provided information)
class_quiz_averages = {
    "Quiz1": 4.26,
    "Quiz2": 3.78,
    "Quiz3": 4.06,
    "Quiz4": 4.45,
    "Quiz5": 4.09,
    "Quiz6": 4.23,
    "Quiz7": 4.00,
    "Quiz8": 2.68,
    "Quiz9": 3.14,
    "Quiz10": 4.86
}

# Calculate individual average for assignments
individual_assignment_score_total = sum(assignment_scores.values())
individual_assignment_max_total = sum(assignment_max_scores.values())

# Calculate individual average for quizzes
individual_quiz_score_total = sum(quiz_scores.values())
individual_quiz_max_total = sum(quiz_max_scores.values())

# Calculate class average for quizzes (sum of class averages)
class_quiz_total = sum(class_quiz_averages.values())
class_quiz_count = len(class_quiz_averages)

# Calculate total possible score for quizzes and assignments
total_possible_assignments = individual_assignment_max_total
total_possible_quizzes = individual_quiz_max_total

# Individual average for assignments and quizzes
individual_assignment_average = (individual_assignment_score_total / total_possible_assignments) * 100
individual_quiz_average = (individual_quiz_score_total / total_possible_quizzes) * 100

# Class average for quizzes
class_quiz_average = (class_quiz_total / class_quiz_count)

# Weighted total average (Assignments = 15%, Quizzes = 10%, Midterm = 25%)
# Assuming midterm score is 12.0 out of 25 (provided by the user)

midterm_score = 12.00
midterm_max = 25

midterm_average = (midterm_score / midterm_max) * 100

# Calculate total weighted average
weighted_assignment_average = individual_assignment_average * 0.15
weighted_quiz_average = individual_quiz_average * 0.10
weighted_midterm_average = midterm_average * 0.25

# Calculate the total weighted average
total_weighted_average = weighted_assignment_average + weighted_quiz_average + weighted_midterm_average

print("Your individual assignment average is:", individual_assignment_average, "%")
print("Your individual quiz average is:", individual_quiz_average, "%")
print("Class average for quizzes is:", class_quiz_average)
print("Your total weighted average is:", total_weighted_average, "%")


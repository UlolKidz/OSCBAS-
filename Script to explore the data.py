import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data = pd.read_csv(r"C:\Users\depri\Downloads\study_performance .csv")  
test_scores = data[['math_score', 'reading_score', 'writing_score']]
print(test_scores.describe())  
print(data['gender'].value_counts())
print(data['race_ethnicity'].value_counts())
print(data['parental_level_of_education'].value_counts())
print(data['lunch'].value_counts())
print(data['test_preparation_course'].value_counts())

sns.histplot(data['math_score'], kde=True, color='blue', label='Math Score')
sns.histplot(data['reading_score'], kde=True, color='green', label='Reading Score')
sns.histplot(data['writing_score'], kde=True, color='red', label='Writing Score')
plt.legend()
plt.title("Distribution of Test Scores")
plt.show()

sns.countplot(y='parental_level_of_education', data=data, palette='viridis')
plt.title("Parental Education Levels")
plt.show()



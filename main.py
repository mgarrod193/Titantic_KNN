import pandas as pd

#reading CSV file
df = pd.read_csv('titanic.csv')

#getting info on the data
df.info()

#Dropping columns not relevent to outcome
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)

#finding out average age of passanger based on class
print(df.groupby('Pclass')['Age'].median())

#fills in the empty cell in the age column with the median age of the class
def fill_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] == 1:
            return 37
        if row['Pclass'] == 2:
            return 29
        return 24
    return row['Age']

#applys the fill_age method to the age column.
df['Age'] = df.apply(fill_age, axis = 1)

#turns sex into a numerical value
def fill_sex(sex):
    if sex == 'male':
        return 0
    return 1

#applys the fill_sex function to the 'Sex' column
df['Sex'] = df['Sex'].apply(fill_sex)


#info for each column checking for any empty values
df.info()

# Step 2. Creating a model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#splitting the data, dropping survived on oone.
X = df.drop('Survived', axis = 1)
Y = df['Survived']

#splitting the data between testing data and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.25)

#scaling the numberical values
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#using KNN alogirthm to train and test.
n_test = 3

#loop increases the value of KNN to find most accurate value.
for i in range(10):

    classifier = KNeighborsClassifier(n_neighbors=n_test)
    classifier.fit(X_train, Y_train)

    Y_pred = classifier.predict(X_test)
    
    print('Accuracy:', accuracy_score(Y_test, Y_pred) * 100)
    print(confusion_matrix(Y_test, Y_pred))
    print('Known nearest neighbours: ', n_test)
    n_test+=2


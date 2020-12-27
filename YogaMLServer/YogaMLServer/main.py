# Import Dependencies
import yaml
from joblib import dump, load
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
import time
from flask import request

global dp


from flask import Flask
app = Flask(__name__)


#global dfx
#People_List =[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
#dfx = DataFrame (People_List,columns=["joint_pain","stomach_pain","acidity","weight_loss","weight_gain","anxiety","breathlessness","indigestion","nausea","fatigue","anxiety","restlessness","loss_of_appetite","back_pain","constipation","fast_heart_rate","obesity","depression"])

@app.route('/')
def hello():
    global dp
    joint_pain = 0 if request.args.get('joint_pain') == None else 1
    stomach_pain = 0 if request.args.get('stomach_pain') == None else 1
    acidity = 0 if request.args.get('acidity') == None else 1
    weight_loss = 0 if request.args.get('weight_loss') == None else 1
    weight_gain = 0 if request.args.get('weight_gain') == None else 1
    anxiety = 0 if request.args.get('anxiety') == None else 1
    breathlessness = 0 if request.args.get('breathlessness') == None else 1
    indigestion = 0 if request.args.get('indigestion') == None else 1
    nausea = 0 if request.args.get('nausea') == None else 1
    fatigue = 0 if request.args.get('fatigue') == None else 1
    restlessness = 0 if request.args.get('restlessness') == None else 1
    loss_of_appetite = 0 if request.args.get('loss_of_appetite') == None else 1
    back_pain = 0 if request.args.get('back_pain') == None else 1
    constipation = 0 if request.args.get('constipation') == None else 1
    fast_heart_rate = 0 if request.args.get('fast_heart_rate') == None else 1
    obesity = 0 if request.args.get('obesity') == None else 1
    depression = 0 if request.args.get('depression') == None else 1
    current_model_name = 'decision_tree'
    #dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    #dp.train_model()
    # Get Model Performance on Test Data
    symptom =[[joint_pain,stomach_pain,acidity,weight_loss,weight_gain,anxiety,restlessness,breathlessness,indigestion,nausea,fatigue,loss_of_appetite,back_pain,constipation,fast_heart_rate,obesity,depression]]
    test_accuracy, classification_report = dp.overchange(current_model_name,symptom)
    return str(test_accuracy[0])




class DiseasePrediction:
    # Initialize and Load the Config File
    def __init__(self, model_name=None):
        # Load Config File
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        
        self.dfx = None
        self.verbose = self.config['verbose']
        # Load Training Data
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # Load Test Data
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # Model Definition
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config['model_save_path']

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-1]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        print(len(train_features.iloc[0]),"dfskjhfkjdshkf")
        assert (len(train_features.iloc[0]) == 17)
        assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Function to Load Test Dataset
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data sanity
        print(len(test_features.iloc[0]),"dfskjhfkjdshkf")
        assert (len(test_features.iloc[0]) == 17)
        print(test_features.shape[0],"atul")
        assert (len(test_labels) == test_features.shape[0])

        if self.verbose:
            print("Length of Test Data: ", df_test.shape)
            print("Test Features: ", test_features.shape)
            print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        #plt.show()
        #plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])

        if self.verbose:
            print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
            print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # Model Selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # ML Model
    def train_model(self):
        # Get the Data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # Training the Model
        classifier = classifier.fit(X_train, y_train)
        # Trained Model Evaluation on Validation Dataset
        confidence = classifier.score(X_val, y_val)
        # Validation Data Prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # Save Trained Model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))


    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        global clf
        print(self.test_features,type(self.test_features))
        print(self.dfx,type(self.dfx),"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
        try:
            # Load Trained Model
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")
        result ="0"
        if test_data is not None:
            result = clf.predict(self.dfx)
            return result
        else:
            result = clf.predict(self.dfx)
        #accuracy = accuracy_score(self.test_labels, result)
        #clf_report = classification_report(self.test_labels, result)
        return result, result
    
    def overchange(self,saved_model_name1, test_data):
        self.dfx = DataFrame (test_data,columns=["joint_pain","stomach_pain","acidity","weight_loss","weight_gain","anxiety","restlessness","breathlessness","indigestion","nausea","fatigue","loss_of_appetite","back_pain","constipation","fast_heart_rate","obesity","depression"])
        print(self.dfx,type(self.dfx),"sdfjhksdfhkjsdhfhkdsfxcvtu ")
        
        accuracy, clf_report = self.make_prediction(saved_model_name=saved_model_name1)
        return accuracy, clf_report



if __name__ == "__main__":
    global dp
    # Model Currently Training
    current_model_name = 'decision_tree'
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    People_List =[[0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0]]
    #dfx = DataFrame (People_List,columns=["joint_pain","stomach_pain","acidity","weight_loss","weight_gain","anxiety","restlessness","breathlessness","indigestion","nausea","fatigue","loss_of_appetite","back_pain","constipation","fast_heart_rate","obesity","depression"])
    #print(dfx,type(dfx))
    #a = pd.DataFrame({"joint_pain":0,"stomach_pain":0,"acidity":0,"weight_loss":0,"weight_gain":0,"anxiety":1,"restlessness":1,"breathlessness":0,"indigestion":0,"nausea":0,"fatigue":0,"loss_of_appetite":0,"back_pain":0,"constipation":0,"fast_heart_rate":0,"obesity":0,"depression":0})
    test_accuracy, classification_report = dp.overchange(current_model_name,People_List)
    print("Model Test Accuracy: ", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)
    app.run()



if __name__ == '__mafdgin__':
    # Model Currently Training
    current_model_name = 'decision_tree'
    # Instantiate the Class
    dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    dp.train_model()
    # Get Model Performance on Test Data
    People_List =[[0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0]]
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    #while 1:
    #    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    #    time.sleep(3)
    #print("Model Test Accuracy: ", test_accuracy)
    #print("Test Data Classification Report: \n", classification_report)
    app.run()

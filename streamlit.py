from google.protobuf.descriptor import EnumValueDescriptor
from numpy import e
from numpy.core.fromnumeric import trace
from pandas.io.stata import precision_loss_doc
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pickle as pkl
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, mean_squared_error, mean_absolute_error, classification_report, PrecisionRecallDisplay, plot_roc_curve, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
import numpy as np

df_cl = pd.read_csv("Test_RandomForest_binary.csv")
df_cl=df_cl.set_index(["USAGE","NAME","PATIENT ID"])
df_cl= df_cl.drop("Predicted_labels",axis=1)


df_main = pd.read_csv("dataset/train.csv")
df_grp = pd.read_csv("train_indexed_grouped.csv")

train = pd.read_csv("Model_train.csv")
train_class = pd.read_csv("model_data_classi.csv")

traini = train.set_index(['USAGE','NAME','PATIENT ID'])





ind = traini.drop(['score','review_stem'],axis=1)
tar = traini.score
(xtr,xte,ytr,yte) = train_test_split(ind,tar,shuffle=True)



model1 = ElasticNet(alpha=1.0,
    l1_ratio=0.24,
    fit_intercept=True,
    normalize=True,
    precompute=False,
    max_iter=1000)
model1.fit(xtr,ytr)
ypred = model1.predict(xte)

model2 = LinearRegression().fit(xtr,ytr)
ypred2 = model2.predict(xte)


indc = train_class.drop(['pos_neg','USAGE','NAME','PATIENT ID'],axis=1)
tarc = train_class.pos_neg
(xtrc,xtec,ytrc,ytec) = train_test_split(indc,tarc,shuffle=True)


model3 = LogisticRegression(penalty='l2',
    dual=False,
    tol=0.0001,
    C=1.0,
    fit_intercept=True,
    intercept_scaling=1,).fit(xtrc,ytrc)
ypred3 = model3.predict(xtec)



model4 = RandomForestClassifier(
    n_estimators=200,
    criterion='entropy',
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=None,
    verbose=0,
    warm_start=False,).fit(xtrc,ytrc)
ypred4 = model4.predict(xtec)


model5 = SGDClassifier(loss='hinge',
    penalty='l2',
    alpha=0.0001,
    l1_ratio=0.15,
    fit_intercept=True,
    max_iter=1000,
    tol=0.001,
    shuffle=True,
    verbose=0,
    epsilon=0.1,
    n_jobs=None,
    random_state=None,
    learning_rate='optimal',
    eta0=0.0,
    power_t=0.5,
    early_stopping=False,
    validation_fraction=0.1,
    n_iter_no_change=5,
    class_weight=None,
    warm_start=False,)
model5.fit(xtrc,ytrc)
ypred5 = model5.predict(xtec)


DrugUsage = {}

for i in train.reset_index().USAGE.unique():
    temp_ls = []
    
    for j in train.reset_index()[train.reset_index().USAGE == i].NAME.unique():
        if np.sum(train.reset_index().NAME == j) >= 10:
            temp_ls.append((j, np.sum(train.reset_index()[train.reset_index().NAME == j].effect) / np.sum(train.reset_index().NAME == j)))

    DrugUsage[i] = pd.DataFrame(data=temp_ls, 
                                columns=['drug',
                                'average_rating']).sort_values(by='average_rating', 
                                ascending=False).reset_index(drop=True)



viz = train.drop(['USAGE','NAME','PATIENT ID'],axis=1)


def main():
    st.title("Drug Review - NLP")
    st.subheader("Coding Assessment")

    menu = ["Brief", "Visualisation", "Prediction"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Prediction":
        st.subheader("Final data")

        st.dataframe(traini)

        st.subheader("Sentiment Classification")
        st.table(traini.drop('review_stem',axis=1).head(5))

        fram = ["Regression", "Classification"]

        swd = st.selectbox("Choose the model",fram)



        if swd == "Regression":
            st.markdown("**Independent** - *Prescription, Effect and Polarity*.  **Dependent** - *Score*")
            st.subheader("Model: ElasticNet Regression")
            st.code("""ElasticNet(alpha=1.0,    l1_ratio=0.24,    fit_intercept=True,    normalize=True,    precompute=False,    max_iter=1000)    """)
            st.write(f"MAE = {mean_absolute_error(yte,ypred)}, MSE = {mean_squared_error(yte,ypred)}")
            st.subheader("Model 2: Simple Linear Regression")
            st.code("LinearRegression(fit_intercept=True,    normalize='deprecated',    copy_X=True,    n_jobs=None,    positive=False,)")
            st.write(f"MAE = {mean_absolute_error(yte,ypred)}, MSE = {mean_squared_error(yte,ypred)}")

        elif swd == "Classification":
            st.write("Classification has done by two separate models, Numeric and NLP. The former only has numeric values for prediction and the latter has only reviews.")
            st.markdown("**Independent** - *Prescription, Review, Effect and Polarity*.  **Dependent** - *pos/neg*")
            sel = ["Machine Learning", "Deep Learning"]

            drs = st.selectbox("select between ML and DL models", sel)

            if drs == "Machine Learning":
                st.subheader("Model 3: Logistic Regession")
                st.code("LogisticRegression(penalty='l2',    dual=False,    tol=0.0001,    C=1.0,    fit_intercept=True,    intercept_scaling=1,)")
                st.write(f"Accuracy = {accuracy_score(ytec,ypred3)}")
                st.write(confusion_matrix(ytec,ypred3))
             

                st.subheader("Model 4: Random Forest Classifier")
                st.code("""
                RandomForestClassifier(n_estimators=200,    criterion='entropy',    max_depth=None,    min_samples_split=2,    min_samples_leaf=1,    min_weight_fraction_leaf=0.0,    max_features='auto',    max_leaf_nodes=None,    min_impurity_decrease=0.0,    bootstrap=True,    oob_score=False,    n_jobs=None,    random_state=None,    verbose=0,    warm_start=False,)                
                """)
                st.write(f"Accuracy = {accuracy_score(ytec,ypred4)}")
                st.write(confusion_matrix(ytec,ypred4))

                st.subheader("Model 5: SGDClassifier ")
                st.code("SGDClassifier(loss='hinge',    penalty='l2',    alpha=0.0001,    l1_ratio=0.15,    fit_intercept=True,    max_iter=1000,    tol=0.001,    shuffle=True,    verbose=0,    epsilon=0.1,    n_jobs=None,    random_state=None,    learning_rate='optimal',    eta0=0.0,    power_t=0.5,    early_stopping=False,    validation_fraction=0.1,    n_iter_no_change=5,    class_weight=None,    warm_start=False,)")
                st.write(f"Accuracy = {accuracy_score(ytec,ypred5)}")
                st.write(confusion_matrix(ytec,ypred5))

            if drs == "Deep Learning":
                st.subheader("Model 4: Simple MLP")
                st.code("""
                                    mod = keras.models.Sequential()

                    mod.add(keras.layers.Dense(500, input_shape=(6000,)))
                    mod.add(keras.layers.BatchNormalization())
                    mod.add(keras.layers.Activation('relu'))
                    mod.add(keras.layers.Dropout(0.9))

                    mod.add(keras.layers.Dense(200))
                    mod.add(keras.layers.BatchNormalization())
                    mod.add(keras.layers.Activation('relu'))
                    mod.add(keras.layers.Dropout(0.7))

                    mod.add(keras.layers.Dense(100, activation='relu'))
                    mod.add(keras.layers.Dense(10, activation='relu'))

                    mod.add(keras.layers.Dense(1, activation='sigmoid'))

                    mod.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                """)

                st.image("simple_summ.png")
                st.image("dfg.png")
                st.write(f"Accuracy of the model is about 75%, which could be more if the scale of the network is to be increased.")

                st.subheader("Model 5: BiDirectional LSTM")
                st.code("""
                                        class RNN_Bidirectional_lstm_Build_Pack():
                            def __init__(self,
                                         input_length,
                                         output_length,
                                         vocab_size,
                                         optimizer,
                                         loss,
                                         metrics,
                                         batch_size,
                                         epochs,
                                         verbose):
        
                                self.input_length =200
                                self.output_length= 200
                                self.vocab_size = 18161
                                self.optimizer = 'adam'
                                self.loss = 'binary_crossentropy'
                                self.metrics = ['acc']
                                self.batch_size = 256
                                self.epochs = 20
                                self.verbose = 1
                   
                                print("Tokenizer object created")
        
                            def build_rnn(self,vocab_size,output_dim, input_dim):

                                model = Sequential([
                                    keras.layers.Embedding(self.vocab_size,output_dim = self.output_length,
                                                          input_length = self.input_length),
                                    keras.layers.BatchNormalization(),
                                    keras.layers.Bidirectional(keras.layers.LSTM(256,return_sequences=True)),
                                    keras.layers.GlobalMaxPool1D(),
                                    keras.layers.Dense(225,activation='relu'),
                                    keras.layers.Dropout(0.3),
                                    keras.layers.Dense(150,activation='relu'),
                                    keras.layers.Dropout(0.2),
                                    keras.layers.Dense(95,activation='relu'),
                                    keras.layers.Dropout(0.2),
                                    keras.layers.Dense(64,activation='relu'),
                                    keras.layers.Dropout(0.1),
                                    keras.layers.Dense(34,activation='relu'),
                                    keras.layers.Dropout(0.1),
                                    keras.layers.Dense(32,activation='relu'),
                                    keras.layers.Dense(output_dim, activation='sigmoid')
                                ])

                                return model
    
                            def Compile_and_Fit(self,rnn_model):
        
                                try:
    
                                    rnn_model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)


                                    rnn_model.fit(x_pad_train, 
                                                            y_train,
                                                            batch_size=self.batch_size,
                                                           epochs=self.epochs,
                                                           verbose= self.verbose)

                                    score = rnn_model.evaluate(x_pad_valid, y_test, verbose=1)

                                    print("Loss:%.3f Accuracy: %.3f" % (score[0], score[1]))

                                    return rnn_model
        
                                except ValueError as Model_Error:
                                    raise(ValueError("Model Compiling Error {}".format(Model_Error)))
                """)
                st.image("summ.png")
                st.image("fgs.png")
                st.write("The Accuracy of this model is already 97% in only about 6 epochs, after which I terminated training. This is so far the best performing model as of yet.")

        




    elif choice == "Visualisation":
        st.subheader("Visual Analytics")
        st.markdown("A brief visual description for the data is given here. For a more detailed report, open the *ipynb notebook* given along. Some of the charts covered here include: \n1. Distribution plot for the various features,\n2. Top Most prescribed/rated drugs by name and use,\n3. Top drugs by with respect to use case.")

        st.subheader("Some plots")
        st.markdown("**SCORE GROUPED BY USAGE**")
        st.line_chart(train.groupby(train.USAGE)['score'].mean().sort_values()[:50])
        st.markdown("**EFFECT GROUPED BY USAGE**")
        
        st.bar_chart(train.groupby('USAGE')['effect'].sum().sort_values()[:50])
        st.markdown("**DRUGS FOR BIRTH CONTROL**")

        st.bar_chart(DrugUsage['Birth Control'].set_index('drug').sort_values(by='average_rating')[:100])
        st.markdown("**DRUGS FOR DEPRESSION**")

        st.bar_chart(DrugUsage['Depression'].set_index('drug').sort_values(by='average_rating')[:100])
        st.markdown("**DRUGS FOR ACNE**")
       
        
        st.bar_chart(DrugUsage['Acne'].set_index('drug').sort_values(by='average_rating')[:100])
        
        st.markdown("**SCORE AND EFFECT GROUPED BY PRESCRIPTIONS**")
      
        st.area_chart(train[['effect','score']].groupby(train.presc).mean().sort_values(by='score')[:50])
        






    else:
        st.subheader("About the data")
        st.dataframe(df_main)
        st.write("The provided training dataset contained 32165 entries of patient record "
                 "\npertaining to 8 of the following columns: "
                 "\n1. Patient ID "
                 "\n2. Name of Drug \n3. Usage \n4. Effect Rating \n5. Review"
                 "\n6. UIC Approval Date"
                 "\n7. No. of Prescriptions"
                 "\n8. Base Score")
        st.subheader("Data grouped by name")
        st.dataframe(df_grp)

        st.write("The data is first cleaned throughout to ensure structural integrity of the analysis. "
                 "The cleaning process includes removing improperly recorded reviews, "
                 "redundant rows and reindexing the columns as follows; use->name->id."
                 "The data is then moved for visual analysis, the complete report of which is given"
                 "in the jupytr notebook. Go to he 'Visualisation tab to get a brief report on the same.")

        st.subheader("Processing the data")
        st.write("Several of the following steps were taken to tokenize the text data into numeric form; "
                 "\n1. Stripping unwanted html characters and punctuations"
                 "\n2. Converting all to lowercase"
                 "\n3. Removing stop words and rare words"
                 "\n4. Stemming and lemmatization"
                 "\n5. Adding polarity"
                 "\n6. Adding the Prediction column as score > 5")
        st.table(df_cl.head(5))

        st.subheader("Modelling the data")
        st.write("""
                 Both machine learning and deep learning models were run for the given data. Models were built for
                 both regression (score prediction) and classification purposes. The ML models that were used were:
                 SLR, ElasticNet regression, SGDClassifier, LogisticReg, RandomForest & Multinomial NaiveBayes.
                 The DL models used were MLP(feedforward), BiDirectional LSTM & transformers. It is observed however,
                 considering execution time, that tradition models based on only numeric data, like logistic 
                 regression and random forest classifier, were generally outperforming complex networks. 
                 Deep learning models never the less performed far better in pure text analysis than any 
                 traditional machine learning model. The best two models for each regression and classification is 
                 given in the 'Prediction' sidebar."
                 """
                 )

        st.subheader("Remarks")
        st.markdown("*FINAL REMARKS*")


if __name__ == '__main__':
    main()




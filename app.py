import pandas as pd
import numpy as np
import emoji 
import tensorflow as tf
import pickle
import keras
import requests
from tensorflow.keras import datasets, layers, models,Input,Model
from shiny import App, render, ui,reactive, ui,run_app
import sklearn 
import zipfile
import json
from tensorflow.keras.layers import TextVectorization


# vectorizer

with zipfile.ZipFile('vectorizer.zip', 'r') as zip_ref:
    zip_ref.extractall()
    
# Load the vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Rebuild the TextVectorization layer
vectorize_layer = TextVectorization(max_tokens=20000, pad_to_max_tokens=True, output_mode='int')
vectorize_layer.set_vocabulary(vocab)



# K means
with open("kmea.pkl","rb") as file:
    kmea=pickle.load(file)
    
    
model = keras.models.load_model('model.h5')

#feature miner
feature_miner=Model(
    inputs=model.inputs,
    outputs=model.get_layer(name="Bidirectional2").output,name="feature_miner"
)

# replacement
strange=pd.read_html(requests.get('https://lhncbc.nlm.nih.gov/LSG/Projects/lvg/current/docs/designDoc/UDF/unicode/DefaultTables/symbolTable.html').content)[-1] #getting the webpage with the table
strange=strange.append(pd.DataFrame({'Unicode':["U+302D"],      #  cast the table into a dataframe
              'Mapped String':["..."],
              "Char":["…"],
              'Unicode Name':["THREE DOTS"]}),ignore_index=True)
s_n=len(strange)
strange=strange.drop(list(strange["Mapped String"]).index("'"))

# text cleaning
def my_clean(text):
    text=" ".join(text.split())
    text=text.lower()
    return text

# emoji removal

def remove_emoji(string):
    return emoji.get_emoji_regexp().sub(u'', string)
# load naive bayes

with open('naivebayes.pkl', 'rb') as f:
    nv = pickle.load(f)
    
AUTOTUNE = tf.data.AUTOTUNE 
app_ui = ui.page_fluid(
            ui.tags.style(
        """
    body {
      background-color: #F5F5F5;
    }
    h3, h4 {
      color: #2F408D;
    }
    #Go{
    background-color:#2F408D;
      color: white;
    }
        """
    ),
    ui.h3("Fake news detection app"),
           ui.tags.br(),
    ui.row(
        ui.column(
            1,),
        ui.column(
            10,
    ui.input_text_area(id="article",label="",placeholder="Enter article ",value="",height='200px',width='1000px'),
        ),
        ui.column(
            1,)),
    ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.input_action_button(id="Go",label="Search",width="200px"),
        ),
        ui.column(
            4,)),
    ui.tags.br(),

        ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.output_ui("txt_title")),
        ui.column(
            4,)),

    ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.output_text("txt_length")),
        ui.column(
            4,)),

    ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.output_text("txt1")),
         ui.column(
            4,)),
        ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.output_text("txt2")),
            ui.column(
            4,)),
     ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.output_text("txt3")),
         ui.column(
            4,)),
     ui.row(
        ui.column(
            4,),
        ui.column(
            4,
    ui.output_text("txt4")),
                 ui.column(
            4,)),
)



def server(input, output, session):
    @output
    @render.ui
    @reactive.event(input.Go)
    def txt_title():
        return ui.tags.h4("Results") 
      
    @output
    @render.text
    @reactive.event(input.Go)
    def txt_length():
        xx = str(input.article())
        xx=len(xx.split())
        return f"Article length: {xx} words"

    
    @reactive.Calc
    @reactive.event(input.Go)
    def text_preps():
        text = input.article()
        text=remove_emoji(text)
        for i in [x for x in range(s_n) if x != 2]:
            text=text.replace(strange["Char"][i],str(strange["Mapped String"][i]))
        text=my_clean(text)
        textdata=tf.data.Dataset.from_tensor_slices([text])
        textdata= textdata.batch(batch_size=64)
        textdata=textdata.map(vectorize_layer).cache().prefetch(buffer_size=AUTOTUNE)
        return textdata

    @reactive.Calc
    def Ann_pred(): 
        textdata = text_preps()
        modelprediction=model.predict(textdata)[0]
        return modelprediction
    
    @reactive.Calc
    def Kmean_pred(): 
        textdata = text_preps()
        kmeans_pred=1-kmea.predict(feature_miner.predict(textdata))
        return kmeans_pred
    
    @reactive.Calc
    def naive_pred(): 
        kmeans_pred = Kmean_pred()
        modelprediction=Ann_pred()
        modelprediction1=round(modelprediction[0])
        final=np.stack((kmeans_pred,np.array([modelprediction1],dtype="int32")), axis=1)
        naive=nv.predict(final)
        return naive
    
    @output
    @render.text
    def txt1():
        modelprediction=Ann_pred()
        xs=round(100*modelprediction[0],4)
        return f"Probability it is fake(based on ANN) : {xs*100}%"

    @output
    @render.text
    def txt2():
        modelprediction=Ann_pred()
        modelprediction1=round(modelprediction[0])
        if modelprediction1==0:
            x="Reliable"
        else:
            x="Fake"
        xx=x
        return f"ANN prediction: {xx}"

    @output
    @render.text
    def txt3():
        kmeans_pred = Kmean_pred()
        if kmeans_pred==0:
            x="Reliable"
        else:
            x="Fake"
        xx=x
        return f"Kmeans prediction: {xx}"

    @output
    @render.text
    def txt4():
        kmeans_pred = Kmean_pred()
        modelprediction=Ann_pred()
        modelprediction1=round(modelprediction[0])
        final=np.stack((kmeans_pred,np.array([modelprediction1],dtype="int32")), axis=1)
        naive_pred=nv.predict(final)
        
        if naive_pred==0:
            x="Reliable"
        else:
            x="Fake"
        xx=x
        return f"weighted prediction: {xx}"

app = App(app_ui, server)

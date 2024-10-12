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
import json
from tensorflow.keras.layers import TextVectorization
import lime
import lime.lime_text


# vectorizer
   
# Load the vocabulary
with open('vocab.json', 'r') as f:
    vocab = json.load(f)

# Rebuild the TextVectorization layer
vectorize_layer = TextVectorization(max_tokens=50000, pad_to_max_tokens=True, output_mode='int')
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
new_row = pd.DataFrame({
    'Unicode': ["U+302D"], 
    'Mapped String': ["..."],
    'Char': ["…"],
    'Unicode Name': ["THREE DOTS"]
})
strange = pd.concat([strange, new_row], ignore_index=True)              
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

def decode_text(indices):
    return " ".join([vectorize_layer.get_vocabulary()[index] for index in indices if index != 0])

def predict_prob(texts):
    sequences = vectorize_layer(tf.constant(texts))
    predictions = model.predict(sequences)
    return predictions

AUTOTUNE = tf.data.AUTOTUNE 

app_ui = ui.page_fluid(
    ui.tags.style(
    """
    body {
        background-color: #F5F5F5;
        font-family: 'Arial', sans-serif;
        color: #333;
    }
    h3, h4 {
        color: #2F408D;
        text-align: center;
    }
    .card {
        background-color: #fff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card-header {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 15px;
    }
    .card-body {
        font-size: 16px;
    }
    #Go {
        background-color: #2F408D;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        transition: background-color 0.3s ease;
    }
    #Go:hover {
        background-color: #3A50B3;
    }
    """
    ),
    ui.h3("Fake News Detection App"),
    ui.row(
        ui.column(2,),
        ui.column(8, ui.input_text_area(id="article", label="", placeholder="Enter article", height='200px', width='1000px')),
    ui.column(2,)
    ),
     ui.row(
         ui.column(4,),
        ui.column(4, ui.input_action_button(id="Go", label="Search", width="200px",align="center")),
    ui.column(4,)
    ),
    ui.output_ui("txt_title"),
    ui.row(
        ui.column(12, 
            ui.tags.div(
                ui.tags.div("Prediction Results", class_="card-header"),
                ui.tags.div(ui.output_text("txt_length"), class_="card-body"),
                ui.tags.div(ui.output_text("txt1"), class_="card-body"),
                ui.tags.div(ui.output_text("txt2"), class_="card-body"),
                ui.tags.div(ui.output_text("txt3"), class_="card-body"),
                ui.tags.div(ui.output_text("txt4"), class_="card-body"),
                class_="card"
            ),
        )),
        ui.row(
        ui.column(12, 
            ui.tags.div(
                ui.tags.div("LIME Explanation", class_="card-header"),
                ui.output_text("lime_output"), # For displaying LIME explanations
                class_="card"
            )
        )
    )
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
        return f"Probability it is fake(based on ANN) : {round(xs,2)}%"

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
    
   # LIME Explanation logic
    @reactive.Calc
    @reactive.event(input.Go)
    def lime_explanation():
        # Prepare the input
        text = input.article()
        textdata = text_preps()
        x_sample = next(iter(textdata))
        
        # Decode the sample text
        decoded_text = decode_text(x_sample[0].numpy())
        
        # LIME explanation
        explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])
        explanation = explainer.explain_instance(decoded_text, predict_prob, num_features=10)
        return explanation.as_html()
    
    @output
    @render.ui
    def lime_output():
        lime_html = lime_explanation()
        return ui.HTML(lime_html)


app = App(app_ui, server)
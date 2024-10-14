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
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.pyplot as plt
from icons import gear_fill,info_circle_fill

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

base_orange = (1, 0.65, 0)  # Lighter orange for positive contributions
base_blue = (0.27, 0.51, 0.71)  # Lighter blue for negative contributions
base_white = (1, 1, 1)  # White for zero contribution

def contribution_to_background_color(word, contribution, max_contribution=1):
    norm_contribution = contribution / max_contribution
    norm_contribution = max(min(norm_contribution, 1), -1)
    if norm_contribution > 0:
        darkened_color = [base_white[i] * (1 - norm_contribution) + base_orange[i] * norm_contribution for i in range(3)]
    elif norm_contribution < 0:
        darkened_color = [base_white[i] * (1 + norm_contribution) + base_blue[i] * -norm_contribution for i in range(3)]
    else:
        darkened_color = base_white
    return f'<span style="background-color:{mcolors.to_hex(darkened_color)}; padding:2px;">{word}</span>'


AUTOTUNE = tf.data.AUTOTUNE 

 
# Define the UI
app_ui = ui.page_fluid(
    ui.tags.div("News Authentication app",class_='title-bar'),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric(id="num_features",label="Number of feature",value=100,min=0,max=None),
            ui.input_numeric(id='num_samples',label="number of samples",value =5000,min=0,max=None),
            ui.input_action_button(id="Go", label="Check", width="200px",align="center")
        ),

        ui.card(
        ui.card_header(
            "Enter Article",
            ui.popover(
                ui.span(
                    info_circle_fill,
                    style="position:absolute; top: 5px; right: 7px;",
                ),
                ui.div(
                    ui.tags.p("This application predicts whether a news article is fake or real based on user input."),
                    ui.tags.p("You can enter the article text and adjust the parameters to see how the predictions change."),
                    ui.tags.p("Use the 'Check' button to get predictions."),
                    ui.tags.p("The models used here were trained with articles longer than 200 characters, so this app works better for such cases."),
                    class_="custom-popover"),
                placement="right",
                id="card_popover",
                
            ),
        ),
        ui.input_text_area(id="article", label="", placeholder="🚀 NASA's #PsycheMission launched on October 13, 2023, to explore asteroid 16 Psyche, believed to be the exposed core of a planetesimal! 🪐 The mission aims to unlock secrets of planetary formation. Psyche is mostly metal and could offer clues about Earth’s core. Stay tuned as the spacecraft journeys over 2 billion miles to reach it by 2029! #SpaceExploration #NASA #Asteroid", height='200px', width='1200px')
    ),    
        ui.layout_columns(
            ui.card(
                ui.card_header("Prediction Results"),
                ui.card_body(
                   ui.layout_column_wrap(
                    ui.card(
                        ui.card_body(
                        ui.output_ui("txt_length"),
                        #ui.output_ui("txt1"),
                        ui.output_ui("txt2"),
                        ui.output_ui("txt3"),
                        ui.output_ui("txt4")),
                        fill=True
                    ),
                     ui.card(
                        ui.card_body(
                            ui.output_plot("bar_fake")
                        ),
                        fill=True
                        ),
                        heights_equal="row",
                        width=1),
                fill=True,
            )),
            ui.card(
                ui.card_header("LIME Explanation"),

                ui.card_body(
                    ui.output_ui("lime_output")
                ),
                fill=True
            ),
            col_widths=[4, 8],
            fill=False
        )
        ),
        ui.tags.div(
        ui.tags.p("© 2024 Fesnic Research Solutions. All rights reserved."),
        ui.tags.div(
            ui.tags.a(
                ui.tags.img(src="https://raw.githubusercontent.com/statisticsguru1/Utility-functions/refs/heads/main/E-learn/images/facebook.svg", alt="Facebook"),
                href="https://www.facebook.com/FesnicResearchSolutions/", target="_blank"
            ),
            ui.tags.a(
                ui.tags.img(src= "https://raw.githubusercontent.com/statisticsguru1/Utility-functions/refs/heads/main/E-learn/images/instagram.svg", alt="Instagram"),
                href="https://www.instagram.com/fesnicresearchsolutions/?hl=en", target="_blank"
            ),
            ui.tags.a(
                ui.tags.img(src="https://raw.githubusercontent.com/statisticsguru1/Utility-functions/refs/heads/main/E-learn/images/linkedin.svg", alt="LinkedIn"),
                href="https://www.linkedin.com/in/festus-nzuma-26580163", target="_blank"
            ),
            ui.tags.a(
                ui.tags.img(src="https://raw.githubusercontent.com/statisticsguru1/Utility-functions/refs/heads/main/E-learn/images/youtube.svg", alt="YouTube"),
                href="https://www.youtube.com/@FesnicResearchSolutions", target="_blank"
            ),
            class_="social-media"
        ),
        class_='custom-foot'
    ),
    ui.include_css(Path(__file__).parent / "styles.css")
)


# Define the server logic
def server(input, output, session):
    @output
    @render.ui
    @reactive.event(input.Go)
    def txt_title():
        return ui.tags.h4("Results") 
      
    @output
    @render.ui
    @reactive.event(input.Go)
    def txt_length():
        xx = str(input.article())
        xx=len(xx.split())
        return ui.HTML(f"<strong>Article length: {xx} words</strong>")

    
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
    @render.ui
    def txt1():
        modelprediction=Ann_pred()
        xs=round(100*modelprediction[0],4)
        return ui.HTML(f"<strong>Prob it is fake fake(based on ANN) :{round(xs,2)}%</strong>")
    # Create the progress bar for "Fake"
    @output
    @render.plot   
    def bar_fake():
        plt.figure(figsize=(3, 3))
        categories = ['Fake', 'Real']
        values = [round(Ann_pred()[0], 4), round(Ann_pred()[1], 4)]
        colors = ['#145da0', '#FF8C00']
        plt.bar(categories, values, color=colors)
        for index, value in enumerate(values):
            plt.text(index, value, str(value), ha='center', va='bottom', fontsize=12)
            plt.title("Probabilities")
            plt.tight_layout()
            plt.show()


    @output
    @render.ui
    def txt2():
        modelprediction=Ann_pred()
        modelprediction1=round(modelprediction[0])
        if modelprediction1==0:
            x="Reliable"
        else:
            x="Fake"
        xx=x
        return ui.HTML(f"<strong>ANN prediction: {xx}</strong>")

    @output
    @render.ui
    def txt3():
        kmeans_pred = Kmean_pred()
        if kmeans_pred==0:
            x="Reliable"
        else:
            x="Fake"
        xx=x
        return ui.HTML(f"<strong>Kmeans prediction: {xx}</strong>")

    @output
    @render.ui
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
        return ui.HTML(f"<strong>weighted prediction: {xx}</strong>")
    
   # LIME Explanation logic
    @reactive.Calc
    @reactive.event(input.Go)
    def lime_explanation():
        # Prepare the input
        textdata = text_preps()
        x_sample = next(iter(textdata))
        # Decode the sample text
        decoded_text = decode_text(x_sample[0])
        # LIME explanation
        explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])
        explanation = explainer.explain_instance(decoded_text, predict_prob, num_features=input.num_features(),num_samples=input.num_samples())
        text_explanation=explanation.as_list()
        max_contribution = max(abs(c) for _, c in text_explanation)
        colored_text = []
        
        for word in input.article().split():
            for explained_word, contribution in text_explanation:
                if word in explained_word:
                    colored_text.append(contribution_to_background_color(word, contribution, max_contribution))
                    break
            else:
                colored_text.append(f'<span style="white-space:nowrap">{word}</span>')
        colored_text_html = " ".join(colored_text)
        return colored_text_html

    
    @output
    @render.ui
    def lime_output():
        return ui.HTML(f'<div style="display: inline-block; white-space: normal;">{lime_explanation()}</div>')


app = App(app_ui, server)

# 📢 Hybrid Fake News Detection System

---

## 📝 Description
This project aims to detect whether a given news article is **fake** or **reliable** using a hybrid machine learning approach. The system combines the strengths of:

- 🔮 **Deep Learning LSTM Classifier**
- 🔍 **K-means Clustering Model**
- 🧠 **Naive Bayes Classifier** for merging predictions.

The model is deployed using **Shiny for Python**, allowing users to:

- Input news articles 📰.
- Receive predictions on authenticity ✅/❌.
- View LIME explanations for interpretability ✨.

---

## ✨ Features
- **Hybrid System**:
  - 🔮 LSTM Classifier for advanced text classification.
  - 📊 K-means Clustering for article grouping.
  - 🧠 Naive Bayes Classifier for combining model predictions.
- **Interactive Interface**: Built with Shiny for Python.
- **Detailed Predictions**:
  - 🔵 Probabilities for "Real" vs "Fake."
  - 🔦 LIME explanations highlighting key phrases influencing predictions.

---

## 🛠️ Technologies Used
- **Shiny for Python**: Interactive user interface.
- **Deep Learning (LSTM)**: Text classification backbone.
- **K-means Clustering**: For unsupervised grouping.
- **Naive Bayes Classifier**: For merging results.
- **LIME**: To explain model decisions.

---

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/FestusNzuma/hybrid-fake-news-detection.git
   ```
2. Install Python dependencies:
   ```bash
   pip install tensorflow shiny lime
   ```
3. Run the Shiny app:
   ```bash
   shiny run --app app.py
   ```

---

## 💻 Usage
1. Launch the app using the command above.
2. Paste or type a news article in the input box 🖊️.
3. Review the output:
   - **Prediction**: Real or Fake 🟢/🔴.
   - **Probabilities**: Confidence levels 📈.
   - **LIME Explanation**: Highlights the words contributing to the prediction 🔦.

---

## 🌐 Live Application
The app is live and accessible here: [Hybrid Fake News Detection System](https://0193ddeb-22a2-5e2b-3079-04bec298aed5.share.connect.posit.cloud/)

---

## 🤝 Contributing
We welcome contributions! Here's how to get started:

1. **Fork the Repository** 📤.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a Pull Request 📬.

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments
Special thanks to the open-source community and contributors for their invaluable resources and support! 🌟

---

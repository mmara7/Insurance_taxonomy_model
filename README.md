# Insurance_taxonomy_model

This project builds a robust company classifier that assigns companies to labels from a predefined file (.xlsx). The solution leverages modern natural language processing (NLP) techniques and semantic embeddings to classify companies based on their descriptions, business tags, and sector/category/niche classification.

To implement a good solution for a classification problem, the first step is to process the data by cleaning and normalize the descriptions. (code/data_processor.py)

To classify I used semantic embeddings to capture the meaning of company descriptions and insurance taxonomy labels. A pre-trained model (all-mpnet-base-v2) from the sentence-transformers library is used to generate embeddings. This model is well-suited for semantic similarity tasks, allowing to compare the similarity between companies and the taxonomy labels.
Compared to a BERT model, for this kind of task is better to use this kind of pre-trained model. 

To make the classification work, the main way to do that is by using similarity search. I used the FAISS library - allows you to search fast the most similar labels based on company's embeddings. 

Project structure:

config/setting.py: Contains model configuration and device settings.

code/data_processor.py: Handles text cleaning and feature extraction.

core/embedding_generator.py: Generates semantic embeddings for the company descriptions and taxonomy labels.

core/model.py: Implements the FAISS-based similarity search and classification logic.

scripts/train.py: Main script to train the model, generate predictions, and visualize results.

utils/device_check.py: Verifies CUDA and hardware configuration.

As any other python program, the dependencies are essentials!! so, make sure you have all you need. I recommend using gpu if possible, so you speed up the process. It's a life saver. If not, feel free to use cpu. 

After running the train.py script the output "companies_with_insurance_labels.csv" will contain the original data with an additional column "insurance_label"

I added a visualisations file to make it easier to understand.

Code Explanation
Data Processor
The DataProcessor class is responsible for cleaning and preparing the input data. It processes various textual features—such as company descriptions, business tags, sector, category, and niche—by removing punctuation, converting text to lowercase, and eliminating unnecessary characters like digits. 
This preprocessing step ensures that the data is in a consistent, normalized format, which is essential for generating meaningful embeddings.

Embedding Generator
In the EmbeddingGenerator class, I used the sentence-transformers library to generate semantic embeddings for the company descriptions and the taxonomy labels. This approach captures the underlying meaning of the text rather than just matching keywords. 
The embeddings are essential for comparing the company data with the insurance taxonomy labels and determining which labels are most relevant to each company.

Insurance Model
The core of the solution is the InsuranceModel class, which utilizes FAISS (Facebook AI Similarity Search) to build an efficient index of the insurance taxonomy labels. 
This index allows for fast similarity searches, where the model compares the company embeddings to the taxonomy embeddings and returns the top relevant labels based on similarity.

Visualization
I incorporated various visualization techniques to evaluate the model’s performance. Functions like plot_label_distribution, plot_label_coverage, and plot_embedding_clusters are used to analyze the distribution of predicted labels, assess how well the model is covering all labels, and visualize how the company embeddings cluster together. 
These visualizations provide insights into the model’s predictions and help identify areas for improvement.

One of the strengths of this approach is its scalability. The use of FAISS allows the solution to handle large datasets efficiently, enabling quick similarity searches even as the data grows.


One of the things to improve might be the limitations of the labels because the model cand struggle with some companies, but it was a good leason to learn : text preprocessing and model trained on text and embedding generation

I wanted to balance the accuracy and efficiency, but the next step for me is to try deep learning-based classifiers to make the model even more accurate and a more complex model. One big step for improvement might be active learning, for additional labels and, of course, fine-tuning. 

Overall, this project gave me a deeper understanding of how to approach text classification tasks. I learned a lot about training models, generating embeddings, and the importance of scalability and interpretability in real-world machine learning applications. While the current solution is effective, there’s always room for improvement, particularly when it comes to handling edge cases and improving accuracy. 
I’m excited to continue learning and refining my skills, particularly in training more sophisticated models and experimenting with different machine learning techniques.

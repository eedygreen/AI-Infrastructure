## Quiz: Analyzing AI Model Output Text
### Scenario:
You are analyzing the output of an AI model from a recent project. You need to answer specific questions about the text using string methods.

### Task:
Answer the following questions about the string variable model_output:

1. What is the length of the string variable model_output?
2. What is the index of the first occurrence of the word 'accuracy' in model_output?
3. What is the index of the last occurrence of the word 'model' in model_output?
4. What is the count of occurrences of the word 'model' in model_output?
5. Tokenize the model_output into individual words.
6. Perform a basic sentiment analysis by counting the number of positive and negative words in the model_output.


```python
model_output = "The AI model achieved an accuracy of 92.5% in the initial tests.\nThe model's performance was consistent across different datasets.\nFurther tuning of the model hyperparameters improved the accuracy to 94%.\nThis model is now ready for deployment in the production environment.\nModel performance will be monitored continuously to ensure it meets the expected standards."

print(model_output)

# Answer the questions
length_of_model_output = len(model_output) # TODO
index_first_accuracy = model_output.index('accuracy', 0) # TODO
index_last_model = model_output.rfind('model') # TODO
count_model = model_output.count('model')# TODO

# Tokenize the model_output into individual words
tokens = model_output.split() # TODO

# Perform a basic sentiment analysis
positive_words = ['achieved', 'consistent', 'improved', 'ready', 'meets']
negative_words = ['monitored']

count_positive = sum(token in tokens for token in positive_words) # TODO
count_negative = sum(token in tokens for token in negative_words) # TODO
print("-"*50)
# Output your answers in descriptive messages
print("The length of the model output string is {} characters.".format(length_of_model_output))
print("The first occurrence of the word 'accuracy' is at index {}.".format(index_first_accuracy))
print("The last occurrence of the word 'model' is at index {}.".format(index_last_model))
print("The word 'model' occurs {} times in the model output.".format(count_model))
print("The tokenized model output is: {}".format(tokens))
print("The number of positive words is: {}".format(count_positive))
print("The number of negative words is: {}".format(count_negative))
```

    The AI model achieved an accuracy of 92.5% in the initial tests.
    The model's performance was consistent across different datasets.
    Further tuning of the model hyperparameters improved the accuracy to 94%.
    This model is now ready for deployment in the production environment.
    Model performance will be monitored continuously to ensure it meets the expected standards.
    --------------------------------------------------
    The length of the model output string is 366 characters.
    The first occurrence of the word 'accuracy' is at index 25.
    The last occurrence of the word 'model' is at index 210.
    The word 'model' occurs 4 times in the model output.
    The tokenized model output is: ['The', 'AI', 'model', 'achieved', 'an', 'accuracy', 'of', '92.5%', 'in', 'the', 'initial', 'tests.', 'The', "model's", 'performance', 'was', 'consistent', 'across', 'different', 'datasets.', 'Further', 'tuning', 'of', 'the', 'model', 'hyperparameters', 'improved', 'the', 'accuracy', 'to', '94%.', 'This', 'model', 'is', 'now', 'ready', 'for', 'deployment', 'in', 'the', 'production', 'environment.', 'Model', 'performance', 'will', 'be', 'monitored', 'continuously', 'to', 'ensure', 'it', 'meets', 'the', 'expected', 'standards.']
    The number of positive words is: 5
    The number of negative words is: 1


**About Sentiment Analysis**

Sentiment analysis is a natural language processing (NLP) technique used to determine the sentiment expressed in a piece of text. It involves classifying text as positive, negative, or neutral based on the words and context. Sentiment analysis is commonly used to analyze customer feedback, social media posts, and product reviews to gauge public opinion and understand how people feel about a particular subject or entity.

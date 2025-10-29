# Create this as data_collector.py first
from datasets import Dataset
import pandas as pd

def create_sample_data():
    human_texts = [
        "The weather is nice today for a walk in the park.",
        "Researchers found interesting results in their latest study.",
        "People enjoy different types of food based on their culture.",
        "The new software update includes several important bug fixes.",
        "Learning new skills takes time and consistent practice."
    ]
    
    ai_texts = [
        "In conclusion, the remarkable advancements have substantially transformed various aspects.",
        "Moreover, it is important to note that these developments offer unprecedented opportunities.",
        "Furthermore, the integration of sophisticated algorithms demonstrates exceptional capability.",
        "Additionally, research has shown that these methodologies provide significant advantages.",
        "Consequently, organizations leveraging these innovations achieve substantial benefits."
    ]
    
    data = {
        'text': human_texts + ai_texts,
        'label': [0] * len(human_texts) + [1] * len(ai_texts)
    }
    
    return Dataset.from_pandas(pd.DataFrame(data))

# Save the dataset
dataset = create_sample_data()
dataset.save_to_disk("./training_data")
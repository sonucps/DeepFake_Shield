import pandas as pd
import os

def create_simple_dataset():
    print("Creating simple training dataset...")
    
    # Human-written texts (label 0)
    human_texts = [
        "The weather is nice today for a walk in the park with friends.",
        "Researchers found interesting results in their latest study on climate change.",
        "People enjoy different types of food based on their culture and background.",
        "The new software update includes several important bug fixes and security improvements.",
        "Learning new skills takes time and consistent practice to master properly.",
        "I went to the store to buy groceries for dinner tonight with my family.",
        "The movie we watched last night had a great storyline and amazing acting.",
        "Students need to study regularly to perform well in their examinations.",
        "My favorite hobby is reading books about history and science fiction.",
        "The team worked hard to complete the project before the deadline."
        "Climate change has become one of the most discussed issues in recent years. People are noticing the effects in their daily lives, from hotter summers to unpredictable rainfall. Governments around the world are trying to balance economic growth with environmental protection."
    ]
    
    # AI-generated texts (label 1)
    ai_texts = [
        "In conclusion, the remarkable advancements have substantially transformed various aspects of modern society.",
        "Moreover, it is important to note that these developments offer unprecedented opportunities for innovation.",
        "Furthermore, the integration of sophisticated algorithms demonstrates exceptional capability in data processing.",
        "Additionally, research has shown that these methodologies provide significant advantages in decision-making.",
        "Consequently, organizations leveraging these technological innovations achieve substantial competitive benefits.",
        "The implementation of these strategies has resulted in considerable improvements across multiple domains.",
        "Notably, the convergence of these technologies has created synergistic effects that enhance overall performance.",
        "Accordingly, the adoption of these approaches has led to measurable enhancements in operational efficiency.",
        "Significantly, these innovations have fundamentally altered traditional paradigms within the industry.",
        "Thus, the utilization of these frameworks has enabled unprecedented scalability and flexibility."
    ]
    
    # Create CSV files (simpler and more compatible)
    train_data = []
    
    # Add human texts
    for text in human_texts:
        train_data.append({'text': text, 'label': 0, 'label_name': 'human'})
    
    # Add AI texts  
    for text in ai_texts:
        train_data.append({'text': text, 'label': 1, 'label_name': 'ai'})
    
    # Save as CSV
    df = pd.DataFrame(train_data)
    df.to_csv('./training_data.csv', index=False)
    
    print(f"✅ Dataset created with {len(df)} samples")
    print(f"   - Human texts: {len(human_texts)}")
    print(f"   - AI texts: {len(ai_texts)}")
    print("✅ Dataset saved to: training_data.csv")
    
    return df

if __name__ == "__main__":
    create_simple_dataset()
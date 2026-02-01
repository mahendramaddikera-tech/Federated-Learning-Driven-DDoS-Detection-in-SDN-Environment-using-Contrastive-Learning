import pandas as pd

def get_samples():
    try:
        # 1. Load your datasets
        print("Loading datasets...")
        features = pd.read_csv('train_features.csv')
        labels = pd.read_csv('train_labels.csv')

        # 2. Key columns required by the App (Indices: 0, 2, 3, 4, 5, 6, 15, 16)
        # Based on your file headers: 
        # 0=dt, 2=src, 3=dst, 4=pktcount, 5=bytecount, 6=dur, 15=Protocol, 16=port_no
        required_cols = features.columns[[0, 2, 3, 4, 5, 6, 15, 16]]
        
        # 3. Combine temporarily to filter by label
        combined = features[required_cols].copy()
        combined['label'] = labels['label']

        # 4. Extract 5 Normal (Label 0) and 5 DDOS (Label 1)
        normal_samples = combined[combined['label'] == 0].head(5)
        ddos_samples = combined[combined['label'] == 1].head(5)

        # 5. Combine and Save
        result = pd.concat([normal_samples, ddos_samples])
        
        output_file = 'test_samples.csv'
        result.to_csv(output_file, index=False)
        
        print(f"\nSUCCESS! Created '{output_file}' with 10 rows (5 Normal, 5 DDOS).")
        print("\nHere is a preview of the data:\n")
        print(result.to_string())

    except FileNotFoundError:
        print("Error: Could not find 'train_features.csv' or 'train_labels.csv'.")
        print("Make sure this script is in the same folder as your dataset files.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    get_samples()

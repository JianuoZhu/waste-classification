import os
import pandas as pd

def generate_annotations(data_dir, output_file='annotations.csv'):
    categories = os.listdir(data_dir)
    data = []
    
    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(category_path, image_name)
                    data.append([image_path, category])
    
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df.to_csv(output_file, index=False)
    print(f"Annotation file created: {output_file}")


generate_annotations('dataset-resized')
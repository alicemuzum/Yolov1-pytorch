import torch 
import os
import pandas as pd
from PIL import Image

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform = None):
        
        self.annotations = pd.read_csv(csv_file) # annotations[0] -> image.jpg       
        self.img_dir = img_dir                   # annotations[1] -> 
        self.label_dir = label_dir
        self.transform = transform
        self.B = B
        self.S = S
        self.C = C
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        """
            Returns specified image as PIL Image and its label as (7,7,30)
        """
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []

        # tüm bboxları her satırda bir boxın verisi olacak sekilde topla
        # örn. [6, 0.73, 0.68, 0.13, 0.20] -> [class_label, x, y, width, height]
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()
                ]

                boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)                       # tranformation olucaksa pytorch kullanılacağı için bu adımda tensora cevir

        if self.transform:
            image, boxes = self.transform(image, boxes)
        
        # (7, 7, 30) boyutunda bir zeros matrix oluştur ve her [i,j,21:25]'i karşılık gelen bboxın x, y, width, heightı ile doldur
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist() # tekrar list e çevir
            class_label = int(class_label)

            i, j = int(self.S * y), int(self.S * x)         # x ve y aslında 0-7 aralığındadır ama 
                                                            # labellarda 0-1 aralığında saklanır
                                                            # bu satırda cell'imizin grid üstündeki x ve y koordinatlarını elde ediyoruz. 
            
            x_cell, y_cell = self.S * x - j, self.S * y - i # örn. 7 * 0.5 - 3 => bize cell'in içindeki koordinatı verir.
                                                            # bu satırda bir cell'in solundan sağına 0'dan 1'e gittiğini düşünürsek
                                                            # bize o cell'ddeki bbox'ın tam yerini söyler.
            
            width_cell, height_cell = (                     # width height değerleri S'ye göre olan değerlerdir
                width * self.S,                             # yani 0-1 aralığındadır ve S ile çarparak bbox'ın
                height * self.S                             # gerçek width ve heightını bulabiliriz
            )
                                
            if label_matrix[i, j, 20] == 0:                 # target label -> [c1, c2, ..., c20, pC, x, y, width, height]

                label_matrix[i, j, 20] = 1                  # [i, j, 20] -> Probabilty of that there is an object
                                                            # bunu 1 yapmamızın sebebi herhalde train setimmizde hep obje olması, ayrıca zaten bir class labelımız var yani
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell] 
                )
                label_matrix[i, j, 21:25] = box_coordinates 
                label_matrix[i, j, class_label] = 1

        return image, label_matrix
    

        
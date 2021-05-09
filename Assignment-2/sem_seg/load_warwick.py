import imageio
import glob 
import numpy as np

def load_warwick():
    
    # Loading raw WARWICK training and testing images and labels  
    train_images = [] 
    train_labels = []
    test_images = []
    test_labels = []
        
    for image_path in sorted(glob.glob('**/WARWICK/Train/' + '/image_*.png', recursive=True)): 
        image = imageio.imread(image_path) 
        train_images.append(image) 
        
    for image_path in sorted(glob.glob('**/WARWICK/Train/' + '/label_*.png', recursive=True)):
        image = imageio.imread(image_path) 
        train_labels.append(image)     

    for image_path in sorted(glob.glob('**/WARWICK/Test/' + '/image_*.png', recursive=True)):
        image = imageio.imread(image_path) 
        test_images.append(image)     
        
    for image_path in sorted(glob.glob('**/WARWICK/Test/' + '/label_*.png', recursive=True)):
        image = imageio.imread(image_path) 
        test_labels.append(image) 
        
    xtrain = np.array(train_images)
    ytrain = np.array(train_labels)
    xtest = np.array(test_images)
    ytest = np.array(test_labels) 
        
    return xtrain, ytrain, xtest, ytest 
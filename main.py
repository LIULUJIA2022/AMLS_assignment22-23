import preprocess
from lab_landmarks import *
from A1.A1_train_test import SVM_A1
from A2.A2_train_test import SVM_A2
from B1.B1_train_test import CNN_B1
from B2.B2_train_test import CNN_B2

# Set the GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


CARTOON_DATASET_DIR = os.path.join("Datasets", "cartoon_set")
CELEB_DATASET_DIR = os.path.join("Datasets", "celeba")
CARTOON_ALT_DATASET_DIR = os.path.join("Datasets", "cartoon_set_test")
CELEB_ALT_DATASET_DIR = os.path.join("Datasets", "celeba_test")



############################################### Task A1 ############################################## 

# Data preprocessing (the validation set will split automatically in scikit-learn cross-validation function)
train_X1, test_X1, train_Y1, test_Y1 = preprocess.data_preprocessing_A1(images_dir, celeba_dir, labels_filename)

# Preprocessing extra test dataset 
extra_test_X1, extra_test_Y1 = preprocess.extra_preprocessing_A1(images_test_dir, celeba_test_dir, labels_test_filename)

# Build SVM model
# Build model object model_A1
model_A1 = SVM_A1()
# Train model based on the training set
acc_A1_train, SVM_A1_clf = model_A1.train(train_X1, train_Y1, test_X1, test_Y1) 
# Test model based on the test set
acc_A1_test = model_A1.test(SVM_A1_clf, extra_test_X1, extra_test_Y1)



############################################## Task A2 ############################################## 

# Data preprocessing (the validation set will split automatically in scikit-learn cross-validation function)
train_X2, test_X2, train_Y2, test_Y2 = preprocess.data_preprocessing_A2(images_dir, celeba_dir, labels_filename)

# Preprocessing extra test dataset 
extra_test_X2, extra_test_Y2 = preprocess.extra_preprocessing_A2(images_test_dir, celeba_test_dir, labels_test_filename)

# Build SVM model
# Build model object model_A2
model_A2 = SVM_A2()
# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A2_train, SVM_A2_clf = model_A2.train(train_X2, train_Y2, test_X2, test_Y2) 
# Test model based on the test set
acc_A2_test = model_A2.test(SVM_A2_clf, extra_test_X2, extra_test_Y2)   



############################################## Task B1 ############################################## 

# Preprocessing training dataset 
train_gen1, valid_gen1, eval_gen1, test_gen1 = preprocess.data_preprocessing_B1(cartoon_images_dir, labels_path)

# Build  model
# Build CNN model object model_B1
model_B1 = CNN_B1()
# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_B1_train, model_path1 = model_B1.train(B1_dir, 5, train_gen1, valid_gen1, eval_gen1)
# Test model based on the test set
acc_B1_test = model_B1.test(model_B1_path, test_gen1)



############################################## Task B2 ##############################################

# Preprocessing training dataset 
train_gen2, valid_gen2, eval_gen2, test_gen2 = preprocess.data_preprocessing_B2(cartoon_images_dir, labels_path)

# Build  model
# Build CNN model object model_B2
model_B2 = CNN_B2()
# Train model based on the training set (you should fine-tune your model based on validation set.)
acc_B2_train, model_path2 = model_B1.train(B2_dir, 5, train_gen2, valid_gen2, eval_gen2)
# Test model based on the test set
acc_B2_test = model_B1.test(model_B2_path, test_gen2)



# ======================== Print out your results with following format:==================================

def print_train_test_acc(task, dct1, dct2):
	print(task + 'train accuracy: ')
	for item, value in dct1.items():
		print('{}: ({})'.format(item, value))

	print(task + 'test accuracy: ')
	for item, value in dct2.items():
		print('{} ({})'.format(item, value))

print_train_test_acc('Task A1 result', acc_A1_train, acc_A1_test)
print_train_test_acc('Task A2 result', acc_A2_train, acc_A2_test)
print_train_test_acc('Task B1 result', acc_B1_train, acc_B1_test)
print_train_test_acc('Task B2 result', acc_B2_train, acc_B2_test)




# defining the class for reading the images dataset
from sklearn.metrics import jaccard_score, precision_score, recall_score,f1_score,confusion_matrix , classification_report
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from model import *

model = SwinDeepLab(
    EncoderConfig, 
    ASPPConfig, 
    DecoderConfig
)

class ImageMaskDatasetNew(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_dir = image_folder
        self.mask_dir = mask_folder
        self.file_names = sorted(os.listdir(self.image_dir))
        self.image_files = os.listdir(image_folder)
        self.mask_files = os.listdir(mask_folder)
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, file_name)
        mask_path = os.path.join(self.mask_dir, file_name)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert("L")
        
        # # Convert mask to binary values (0s and 1s)
        # mask = np.array(mask)
        # mask = np.where(mask > 0, 1, 0)
        # mask = Image.fromarray(mask)

        # Apply the transformations to the input image and segmentation mask
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Normalize the input image
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        return image, mask, file_name
    

testval_image_folder = 'C:/Users/rajes/OneDrive/Desktop/dataset/data/test_split/batch/images/'

testval_mask_folder = 'C:/Users/rajes/OneDrive/Desktop/dataset/data/test_split/batch/masks/'
# Create the dataset
testval_dataset = ImageMaskDatasetNew(testval_image_folder, testval_mask_folder, transform=transform)
#val_dataset = ImageMaskDataset(val_image_folder, val_mask_folder, transform=transform)
# Create the DataLoader
batch_size = 4
testval_loader = DataLoader(testval_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Initialize our model and we are loading the weights from checkpoint path
checkpoint = torch.load("C:/Users/rajes/trans_deeplab_new_folder/best_model_weights.pth") # Load the checkpoint file
model.load_state_dict(checkpoint['model_state_dict']) # Load the weights from the checkpoint file to your model

# modified code
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




# Testing loop 




torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
# Define the binary cross-entropy loss function
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001,weight_decay=0.01)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.01)
model.to(device)

# saving files in save_dir
save_dir = 'C:/Users/rajes/OneDrive/Desktop/dataset/results_transdeeplab/best_weights/batch/'
# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.eval()
test_losses = []
test_accs = []
test_loss = 0.0
correct_test =0
total_test = 0
with torch.no_grad():
    jaccard_scores = []
    dice_scores = []
    precision_scores = []
    recall_scores = []
    y_true_all=np.empty(0,dtype=np.int32)
    y_pred_all=np.empty(0,dtype=np.int32)
    for batch_idx, (input_batch, mask_batch, file_names) in enumerate(testval_loader): # Include the file names in the test_loader
        input_batch = input_batch.to(device)
        mask_batch = mask_batch.to(device)
        output_batch = model(input_batch)
        output_batch = torch.sigmoid(output_batch)
        output_batch = (output_batch > 0.5).float()
        predicted = output_batch
        #print(output_batch)

        output_np1 = output_batch.cpu().numpy().astype(int) 
        mask_np1 = mask_batch.cpu().numpy().astype(int) 
        labels=mask_batch
        loss = criterion(output_batch, mask_batch)
        test_loss += loss.item()
        correct_test += (predicted == labels).sum().item()
        total_test += labels.numel()
        #print(mask_batch.shape)
        for i in range(output_batch.shape[0]):
            output = output_batch[i].squeeze()
            mask = mask_batch[i].squeeze()
            #print(mask)
            output_np = output.cpu().numpy().astype(int) 
            mask_np = mask.cpu().numpy().astype(int) 
            jaccard = jaccard_score(mask_np.flatten(), output_np.flatten(), labels=[1], average='binary')
            #dice = dice_score(mask_np.flatten(), output_np.flatten())
            #dice = 2 * np.sum(mask_np * output_np) / (np.sum(mask_np) + np.sum(output_np))
            dice = f1_score(mask_np.flatten(), output_np.flatten(), labels=[1], average='binary')
            precision = precision_score(mask_np.flatten(), output_np.flatten(), labels=[1], average='binary')
            recall = recall_score(mask_np.flatten(), output_np.flatten(), labels=[1], average='binary')
            jaccard_scores.append(jaccard)
            dice_scores.append(dice)
            precision_scores.append(precision)
            recall_scores.append(recall)

            
            # Save the predicted image with its file name
            output_img = output_batch[i].squeeze().cpu().numpy()
            #print(output_img)
            output_img = Image.fromarray(output_img * 255.0).convert('L')
            output_file_name = file_names[i].split('/')[-1] # Get the file name from the full path
            output_file_path = os.path.join(save_dir, output_file_name)
            output_img.save(output_file_path)
        y_true_all=np.concatenate((y_true_all,mask_np1.flatten()))
        y_pred_all=np.concatenate((y_pred_all,output_np1.flatten()))
        test_loss /= len(testval_loader)
        test_acc = correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)


        # jaccard_scores = TP / (TP + FP + FN)
        # mean_jaccard_score = np.mean(jaccard_scores)

# print the Jaccard score for each class and the mean Jaccard score

        # Calculate average scores for label 1
        # jaccard_score_label_1 = np.mean([score for score in jaccard_scores if score > 0])
        # dice_score_label_1 = np.mean([score for score in dice_scores if score > 0])
        # precision_score_label_1 = np.mean([score for score in precision_scores if score > 0])
        # recall_score_label_1 = np.mean([score for score in recall_scores if score > 0])
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        axes[0].imshow(input_batch[i].permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Image")
        axes[1].imshow(mask_batch[i].squeeze().cpu().numpy(), cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[2].imshow(output_batch[i].squeeze().cpu().numpy(), cmap='gray')
        axes[2].set_title("Predicted Mask")
        plt.savefig(os.path.join(save_dir, f'image_{i}.png'))


    confusion_matrix_score = confusion_matrix(y_true_all, y_pred_all , labels = [0,1])
    classification_report_score = classification_report(y_true_all, y_pred_all , labels=[0,1])
    print("confusion matrix",confusion_matrix_score)
    print("classification report",classification_report_score)  
    num_classes = confusion_matrix_score.shape[0]
    TP = np.diag(confusion_matrix_score)
    FP = np.sum(confusion_matrix_score, axis=0) - TP
    FN = np.sum(confusion_matrix_score, axis=1) - TP



    # print('Jaccard scores:', jaccard_scores)
    # print('Mean Jaccard score:', mean_jaccard_score)
    """ Metrics values """
    #score = [s[1:]for s in SCORE]
    print("recall score",recall_score(y_true_all,y_pred_all,labels=[1],average="binary"))
    print("Precision score",precision_score(y_true_all,y_pred_all,labels=[1],average="binary"))
    print("jaccard score",jaccard_score(y_true_all,y_pred_all,labels=[1],average="binary"))
    print("f1_score",f1_score(y_true_all,y_pred_all,labels=[1],average="binary"))

       


    # avg_jaccard = np.mean(jaccard_scores)
    # avg_dice = np.mean(dice_scores)
    # avg_precision = np.mean(precision_scores)
    # avg_recall = np.mean(recall_scores)


# average scores
avg_jaccard = np.mean(jaccard_scores)
avg_dice = np.mean(dice_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)

print('Jaccard score: {:.4f}, Dice score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(avg_jaccard, avg_dice, avg_precision, avg_recall))


# plot accuracy and loss curves for all epochs so far
# plot test loss and accuracy curves
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))
ax.plot(test_losses, label='Test Loss')
ax.plot(test_accs, label='Test Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss/Accuracy')
ax.set_title('Test Loss and Accuracy')
ax.legend()
plt.show()
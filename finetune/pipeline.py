from torch.optim import AdamW
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
import logging
import matplotlib.pyplot as plt
import uuid
from metrics import DocMetric
from dataset import APIDocumentDataset
from models import CustomDebertaClassifier
import pandas as pd
import os


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def train(
    model,
    device,
    train_dataloader,
    val_dataloader,
    lr = 5e-5,
    num_epochs = 10,
    visualize_loss = True,
    label_cnt = None,
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    
    tolerance = 2
    if label_cnt:
        total_cnt = sum(label_cnt)
        class_weights = [total_cnt / class_count for class_count in label_cnt]
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        loss_fn = CrossEntropyLoss(weight=class_weights_tensor)
    else:
        loss_fn = CrossEntropyLoss()
    
    for epoch in tqdm(range(num_epochs), desc='Epochs', unit='epoch'):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, unit='batch'):
            batch = {k: v.to(device) if k != 'api_doc' else v for k, v in batch.items() }
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
            loss = loss_fn(outputs, batch['labels'])
            total_train_loss += loss.item()
            logger.info(f'Loss: {loss.item()}')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")
        
        if epoch % 3 == 2 or epoch == num_epochs-1:
            model.eval()
            total_val_loss = 0
            for batch in tqdm(val_dataloader, desc='Batches', leave=False, unit='batch'): 
                batch = {k: v.to(device) if k != 'api_doc' else v for k, v in batch.items() }
                with torch.no_grad():
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    
                loss = loss_fn(outputs, batch['labels'])
                total_val_loss += loss.item()
        
            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
            
            if val_losses:
                if avg_val_loss >= val_losses[-1]:
                    tolerance -= 1
                    if tolerance == 0:
                        logger.info("Losses no longer decrease, early stop training!")
                        break
            
        
    
    if visualize_loss:
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        _uuid = uuid.uuid4()
        plt.savefig(f"training_loss_plot_v2_{_uuid}.png")
    
    return model


def eval(
    trained_model,
    device,
    test_dataloader,
):
    trained_model.eval()
    
    doc_metric = DocMetric(num_classes=5)
    
    input_api_docs = []
    gt_labels = []
    pred_labels = []
    
    for batch in tqdm(test_dataloader, desc='Batches', leave=False, unit='batch'):
        batch_inputs = {k: v.to(device) if k != 'api_doc' else v for k, v in batch.items() }
        with torch.no_grad():
            outputs = trained_model(input_ids=batch_inputs['input_ids'], attention_mask=batch_inputs['attention_mask'])

        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        doc_metric.accummulate(predicted_labels, batch_inputs['labels'])
        
        input_api_docs.extend(batch_inputs['api_doc'])
        gt_labels.append(batch_inputs['labels'].item())
        pred_labels.append(predicted_labels.item())
    
    doc_metric.log()
    output_df = pd.DataFrame(
        {
            'api_doc': input_api_docs,
            'gt_labels': gt_labels,
            'pred_labels': pred_labels,
        }
    )
    output_df.to_csv('result.csv', header=True, index=False)
    logger.info("Evaluation complete!")
    
    

def run():
    num_labels = 5  # Number of classes
    num_layers = 2  # Number of additional layers
    device = 'mps'  # Change to 'cuda' if using an NVIDIA GPU; if use Apple M1/2, use 'mps'
    load_checkpoint = False

    # Model Initialization
    trained_model = CustomDebertaClassifier(num_labels=num_labels, num_layers=num_layers, device=device)
    trained_model.to(device)
    
    checkpoint_path = 'model_state_dict_v2.pth'
    if load_checkpoint and os.path.exists(checkpoint_path):
        trained_model.load_state_dict(torch.load(checkpoint_path))

    else:
        # Train Model
        train_dataloader, val_dataloader, label_count = APIDocumentDataset.get_train_dataloaders()
        label_count = list(label_count.values())
        
        trained_model = train(trained_model, device, train_dataloader, val_dataloader, label_cnt=label_count)
        torch.save(trained_model.state_dict(), 'model_state_dict_v2.pth')
    
    # Evaluation:
    test_dataloader = APIDocumentDataset.get_test_dataloaders()
    eval(trained_model, device, test_dataloader)
    

if __name__ == "__main__":
    run()

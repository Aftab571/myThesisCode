from sys import settrace
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import gif
from typing import Dict, List, Union

from json.tool import main
from memory_profiler import profile
from dataProcessing import getPreprocessData

import pandas as pd
from torch.nn import ReLU
from neo4j import GraphDatabase

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
from statistics import mean
import plotly.express as px
from sklearn import preprocessing
import coloredlogs, logging
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
import torch_sparse
from torch_geometric.loader import HGTLoader, NeighborLoader,ImbalancedSampler
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, MetaPath2Vec
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv,GATv2Conv, Linear, SuperGATConv,HANConv, HEATConv
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear, HeteroLinear
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm
import time
import wandb
from IPython.display import Image
import os
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from random import randint
from sentence_transformers import SentenceTransformer
import statistics
from comet_ml import Experiment
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Create an experiment with your api key
experiment = Experiment(
    api_key="yO67iXhjD5FRQH0uKNVOkpCuq",
    project_name="MasterThesis",
    workspace="aftab571",
)

coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

pd.options.mode.chained_assignment = None  # default='warn'

torch.cuda.empty_cache()

mylogs = logging.getLogger(__name__)

num_of_neg_samples= 2000
num_of_pos_samples= 2000

seed = 10
data = None
print(seed)



class HEAT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, params):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = HEATConv(-1, out_channels,num_node_types= params['num_node_types'], num_edge_types= params['num_edge_types'], edge_type_emb_dim= params['edge_type_emb_dim'], edge_dim= params['edge_dim'], edge_attr_emb_dim= params['edge_attr_emb_dim'])  


    def forward(self, x, edge_index,node_type,edge_type, edge_attr):
        x = self.conv1(x, edge_index,node_type,edge_type, edge_attr)
        return x


def create_train_val_test_mask(df):
    X = df.iloc[:,df.columns != 'label']
    Y = df['label']
    mask =[]
    X_train_complete, X_test, y_train_complete, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.1,random_state=seed,stratify=Y)
    test_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    test_mask[X_test.index] = True
    skf = StratifiedKFold(n_splits=2,random_state=10, shuffle=True)
    for train_index, val_index in skf.split(X_train_complete, y_train_complete):
        #print("Train_complete:", train_index, "TEST:", test_index)
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = Y[train_index], Y[val_index]

        #X_train, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.2,random_state=1,stratify=Y)
        train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
        val_mask = torch.zeros(df.shape[0], dtype=torch.bool)

        print("y_train: ",Counter(y_train).values()) 
        print("y_val: ",Counter(y_val).values()) 
    

        train_mask[X_train.index] = True
        val_mask[X_val.index] = True

        conf_df = pd.DataFrame()
        conf_df['admmision_id']= X_test['admmision_id']
        conf_df['actual']= y_test

        obj={
            'train_mask_set': train_mask,
            'val_mask_set' :  val_mask,
            'admission': conf_df
        }
        mask.append(obj)
    #print(mask)

    return mask,test_mask
  
 
    # X_train_complete, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.1,random_state=seed,stratify=Y)
    # X_train, X_val, y_trainval, y_testval = train_test_split(X_train_complete,y_train, test_size=0.1,random_state=seed,stratify=y_train)
    
    # print("y_train: ",Counter(y_train).values()) 
    # print("y_test: ",Counter(y_test).values()) 
    # print("y_trainval: ",Counter(y_trainval).values()) 
    # print("y_testval: ",Counter(y_testval).values()) 


    # train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    # test_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    # val_mask = torch.zeros(df.shape[0], dtype=torch.bool)

    # train_mask[X.index] = True
    # #train_mask[X_train.index] = True
    # test_mask[X_test.index] = True
    # val_mask[X_val.index] = True


    # conf_df = pd.DataFrame()
    # conf_df['admmision_id']= X_test['admmision_id']
    # conf_df['actual']= y_test



    # return train_mask,val_mask,test_mask,conf_df

# def train(model,optimizer,criterion,data):
#       model.train()
#       optimizer.zero_grad()  # Clear gradients.
#       out = model(data.x_dict, data.edge_index_dict)  # Perform a single forward pass.
#       mask = data['Admission'].train_mask
#       loss = criterion(out['Admission'][mask], data['Admission'].y[mask])  # Compute the loss solely based on the training nodes. ['Admission']
#       #print(out['Admission'][mask].shape)
#       #print(data['Admission'].y[mask].shape)
#       loss.backward()  # Derive gradients.
#       optimizer.step()  # Update parameters based on gradients.
#       return loss


def train(model,optimizer,criterion,data,train_loader,device):
    model.train()

    total_examples = total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        node_types = {k: v for v, k in enumerate(batch.node_types)}
        edge_types = {k: v for v, k in enumerate(batch.edge_types)}
        # node_types = torch.tensor([batch.node_types.index(l) for l in batch.node_types]).to(device)
        # edge_types = torch.tensor([batch.edge_types.index(l) for l in batch.edge_types]).to(device)
        batch_size = batch['Admission'].batch_size
        out = model(batch.x_dict, batch.edge_index_dict,node_types,edge_types,batch.edge_attr_dict)
        loss = F.cross_entropy(out['Admission'][:batch_size],
                               batch['Admission'].y[:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
    
    loss_val= total_loss / total_examples
   
    return loss_val, out


def test(model,optimizer,criterion,mask,data,device):
      model.eval()
      node_types = torch.tensor([data.node_types.index(l) for l in data.node_types]).to(device)
      edge_types = torch.tensor([data.edge_types.index(l) for l in data.edge_types]).to(device)
      out = model(data.x_dict, data.edge_index_dict,node_types,edge_types, data.edge_attr_dict)
      pred = out['Admission'].argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data['Admission'].y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc,pred

def gnn_model_summary(model):
    
    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0] 
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)
    for parameter in model.parameters(): 
        print(parameter)



#@profile
def main():
    model = None
    prev_mask = None
    global dataset

    heatmaps=[]
    st_time_nodes = time.time()

    df_admission,df_diagnosis,df_drugs,df_labs,df_vitals,df_diagnosis_features,df_demo = getPreprocessData('Graph',edge_merge=False,grp_aggr='mean')

    end_time = time.time()

    print("Time for Preprocessing data (Graph): ",end_time-st_time_nodes)

    mask_list,test_mask = create_train_val_test_mask(df_admission)
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    for aggr in ['sum']:
        print(aggr)
        
        for i in mask_list:
            data = HeteroData()
            

            data['Admission'].x = torch.tensor(df_admission[['gender','age']].values, dtype = torch.float).to(device)  #'ethnicity','marital','religion','gender','age'
            data['Admission'].y =  torch.tensor(df_admission['label'].values, dtype = torch.long).to(device)
            data['Admission'].train_mask = i['train_mask_set'].to(device)
            data['Admission'].val_mask = i['val_mask_set'].to(device)
            data['Admission'].test_mask = test_mask
            data['Labs'].x = torch.tensor(df_labs[['fluid','category']].values, dtype = torch.float).to(device) #df_labs.loc[:, ~df_labs.columns.isin(['hadm_id','lab_name'])].values.tolist()


            data['Admission', 'has_labs', 'Labs'].edge_index = torch.tensor(df_labs[['adm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            data['Admission', 'has_labs', 'Labs'].edge_attr  = torch.tensor(df_labs[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            data['Labs', 'rev_has_labs', 'Admission'].edge_index = torch.tensor(df_labs[['index_col','adm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            data['Labs', 'rev_has_labs', 'Admission'].edge_attr  = torch.tensor(df_labs[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)


            # data['Vitals'].x = torch.tensor(df_vitals[['name']].values, dtype = torch.float).to(device)
            # data['Admission', 'has_vitals', 'Vitals'].edge_index = torch.tensor(df_vitals[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Admission', 'has_vitals', 'Vitals'].edge_attr  = torch.tensor(df_vitals[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # data['Vitals', 'rev_has_vitals', 'Admission'].edge_index = torch.tensor(df_vitals[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Vitals', 'rev_has_vitals', 'Admission'].edge_attr  = torch.tensor(df_vitals[['value']].values.tolist(), dtype=torch.float).contiguous().to(device)

            # # print(df_drugs[['dosage_val']].values)
            # data['Drugs'].x = torch.tensor(df_drugs[['drug_name']].values, dtype = torch.float).to(device)
            # data['Admission', 'has_drugs', 'Drugs'].edge_index = torch.tensor(df_drugs[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Admission', 'has_drugs', 'Drugs'].edge_attr  = torch.tensor(df_drugs[['dosage_val']].values.tolist(), dtype=torch.float).t().contiguous().to(device)

            # data['Drugs', 'rev_has_drugs', 'Admission'].edge_index = torch.tensor(df_drugs[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Drugs', 'rev_has_drugs', 'Admission'].edge_attr  = torch.tensor(df_drugs[['dosage_val']].values.tolist(), dtype=torch.float).t().contiguous().to(device)

            # # #df_diagnosis.iloc[:,4:].drop('index_col',axis=1).values
            # data['Diagnosis'].x = torch.tensor(df_diagnosis_features.values,dtype = torch.float).to(device)
            # data['Admission', 'has_diagnosis', 'Diagnosis'].edge_index = torch.tensor(df_diagnosis[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Diagnosis', 'rev_has_diagnosis', 'Admission'].edge_index = torch.tensor(df_diagnosis[['index_col','hadm_id']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            
            # data['Demography'].x = torch.tensor(df_demo[['atype']].values.tolist(),dtype = torch.float).to(device)
            # data['Admission', 'has_same_demo', 'Demography'].edge_index = torch.tensor(df_demo[['start','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            # data['Demography', 'rev_same_demo', 'Admission'].edge_index = torch.tensor(df_demo[['index_col','start']].values.tolist(), dtype=torch.long).t().contiguous().to(device)

            data.num_node_features = 3
            data.num_classes = len(df_labs['label'].unique())
            #data = T.ToUndirected()(data.to(device))
            #data = T.NormalizeFeatures()(data.to(device))
            #data = T.RandomNodeSplit()(data)
            dataset = data.to(device)

            data = dataset.to(device)
            if not os.path.exists('MIMICDataObj.pt'):
                torch.save(data,'MIMICDataObj.pt')
            # train_loader = NeighborLoader(
            #     data,
            #     # Sample 15 neighbors for each node and each edge type for 2 iterations:
            #     num_neighbors=[4] * 2,
            #     # Use a batch size of 128 for sampling training nodes of type "paper":
            #     batch_size=8,
            #     input_nodes=('Admission', data['Admission'].train_mask),
            # )
            # batch = next(iter(train_loader)) 
            sampler = ImbalancedSampler(data['Admission'].y, input_nodes=data['Admission'].train_mask)  #, input_nodes=data['Admission'].train_mask
            # print(data.edge_types)
            loader = NeighborLoader(data, input_nodes=('Admission', data['Admission'].train_mask), batch_size=256,
                             num_neighbors={key: [30] * 2 for key in data.edge_types})  #sampler=sampler
            
            if data:
                print(data)
                wandb.init(project="test-project", entity="master-thesis-luffy07")
                if model is not None:
                    model = model
                else:
                    params={
                        'num_node_types':len(data.node_types),
                        'num_edge_types':len(data.edge_types),
                        'edge_type_emb_dim':2,
                        'edge_dim':1,
                        'edge_attr_emb_dim':2

                    }
                    model = HEAT(hidden_channels=32,out_channels=data.num_classes,params=params)
                    model = model.to(device)
                    print(model)
                    model = to_hetero(model, data.metadata(), aggr=aggr).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                    criterion = torch.nn.CrossEntropyLoss().to(device)  #weight=torch.tensor([0.15, 0.85])
                # criterion =  FocalLoss(mode="binary", alpha=0.25, gamma=2)
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
                for epoch in range(1, 101):
                    loss,out = train(model,optimizer,criterion,data,loader,device)
                    wandb.log({"Training loss": loss})
                    experiment.log_metric("Training loss",loss,step=epoch)
                    train_acc,pred_train = test(model,optimizer,criterion,data['Admission'].train_mask,data,device)

                    val_acc,pred_val = test(model,optimizer,criterion,data['Admission'].val_mask,data,device)
                    test_acc,pred_test = test(model,optimizer,criterion,data['Admission'].test_mask,data,device)
                    wandb.log({'Train_acc':train_acc,'Validation_acc':val_acc,'Test_acc':test_acc})
                    experiment.log_metric("Train Accuracy",train_acc,step=epoch)
                    experiment.log_metric("Validation Accuracy",val_acc,step=epoch)
                    experiment.log_metric("Test Accuracy",test_acc,step=epoch)
                    #if epoch%100==0:
                    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
                        #print(scheduler.get_last_lr())
                    #scheduler.step()
                #gnn_model_summary(model)
                
                mask_train = data['Admission'].train_mask
                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_train], pred_train[mask_train].cpu())
                print("train cfm: ",cf_matrix)

                mask_val = data['Admission'].val_mask
                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_val], pred_train[mask_val].cpu())
                print("Validation cfm: ",cf_matrix)


                mask_test = data['Admission'].test_mask
                cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu())
                print("test cfm: ",cf_matrix)
                experiment.log_confusion_matrix(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu(),title="SAGE")
                sensitivity = cf_matrix[0,0]/(cf_matrix[0,0]+cf_matrix[0,1])
                print('Sensitivity : ', sensitivity )
                experiment.log_metric("SAGE Sensitivity", sensitivity)

                specificity = cf_matrix[1,1]/(cf_matrix[1,0]+cf_matrix[1,1])
                print('Specificity : ', specificity)
                experiment.log_metric("SAGE Specificity", specificity)

                # explainer = GNNExplainer(model, epochs=200, return_type='log_prob')
                # node_idx = 10
                # node_feat_mask, edge_mask = explainer.explain_node(node_idx, x, edge_index,
                #                                                 edge_weight=edge_weight)
                # ax, G = explainer.visualize_subgraph(node_idx, edge_index, edge_mask, y=data.y)
                # plt.show()

                ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt= '.3g')

                ax.set_title('SAGE Confusion Matrix\n\n')
                ax.set_xlabel('\nPredicted Values')
                ax.set_ylabel('Actual Values ')

                ## Ticket labels - List must be in alphabetical order
                ax.xaxis.set_ticklabels(['Survived','Died'])
                ax.yaxis.set_ticklabels(['Survived','Died'])

                ## Display the visualization of the Confusion Matrix.
                plt.savefig('SAGE_conf.png', dpi=400)
                plt.show()
        
    
        
def seed_everything(seed=seed):                                                  
    #random.seed(seed)                                                            
    torch.manual_seed(seed)                                                      
    torch.cuda.manual_seed_all(seed)                                             
    np.random.seed(seed)                                                         
    os.environ['PYTHONHASHSEED'] = str(seed)                                     
    torch.backends.cudnn.deterministic = True                                    
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything()
    main()
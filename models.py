'''
Groupe all models and functions to acompain trainig and results analysis. 
'''
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics 
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

# from joblib.externals.loky.backend.context import get_context
from hydra.utils import instantiate
from omegaconf import DictConfig,OmegaConf

import myServices as ms

#### main classes

class implementRandomForestCalssifier():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklear
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self,trainDataSet, targetCol, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGrid(gridArgs)
        self.x_train,self.y_train= ms.importDataSet(trainDataSet, targetCol)
        self.y_train = np.array(self.y_train).ravel()
        print(self.x_train.head())
        print("Train balance")
        _,l = listClassCountPercent(self.y_train)
        print(l)
        self.rfClassifier = implementRandomForestCalssifier.createModelClassifier(self)
    
    def createModelClassifier(self):
        estimator = RandomForestClassifier(criterion='entropy', random_state = self.seedRF)
        # Create the random search model
        rs = GridSearchCV(estimator, 
                        param_grid = self.paramGrid, 
                        n_jobs = -1,
                        scoring = 'accuracy',
                        cv = 3,  # NOTE: in this configurtion StratifiedKfold is used by SckitLearn  
                        verbose = 5, 
                        )           
        return rs
    
    def fitRFClassifierGSearch(self):
        self.rfClassifier.fit(self.x_train, self.y_train)
        best_estimator = self.rfClassifier.best_estimator_
        best_params = self.rfClassifier.best_params_
        print(f"The best parameters are: {best_params}")
        return best_estimator, best_params  

class implementOneVsRestClassifier():
    '''
    Class implementing all necessary steps for a ranom multiclass classification with RF Classifier
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self, dataSet, targetCol, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGridOneVsRest(gridArgs)
        self.x_train,Y = ms.importDataSet(dataSet, targetCol)
        Y = np.array(Y).ravel()
        self.y_train, self.labels = pd.factorize(Y) # See: https://pandas.pydata.org/docs/reference/api/pandas.factorize.html
        # self.x_train,self.x_validation,self.y_train, self.y_validation = train_test_split(X,Y_factorized, test_size = splitProportion) 
        print(self.x_train.head())
        print("Train balance")
        listClassCountPercent(self.y_train)
        # print("Validation balance")
        # printArrayBalance(self.y_validation)
        self.OneVsRestClassifier = implementOneVsRestClassifier.createModel(self)

    def createModel(self):
        estimator = RandomForestClassifier(criterion='entropy', random_state = self.seedRF, bootstrap = False) 
        model_to_set = OneVsRestClassifier(estimator)     
        # Create the random search model
        rs = GridSearchCV(model_to_set, 
                        param_grid = self.paramGrid, 
                        scoring = 'accuracy',
                        cv = 3,  # NOTE: in this configurtion StratifiedKfold is used by SckitLearn 
                        )           
        return rs
    
    def fitOneVsRestClassifierGSearch(self):
        self.OneVsRestClassifier.fit(self.x_train, self.y_train)
        best_params = self.OneVsRestClassifier.get_params
        model = self.OneVsRestClassifier.best_estimator_
        print(f"The best parameters are: {best_params}")
        return model, best_params  

class implementRandomForestRegressor():
    '''
    Class implementing all necessary steps for a ranom Forest 
    regression with sklearnpare
    @imput:
      @ dataset: The full path to a *.csv file containing the dataSet.
      @ targetCol: The name of the column in the dataSet containig the target values.
      @ splitProportion: The proportion for the testing set creation.
    '''
    def __init__(self, dataSet, targetCol, gridArgs):
        self.seedRF = 50
        self.paramGrid = createSearshGrid(gridArgs)
        self.x_train, self.y_train= ms.importDataSet(dataSet, targetCol)
        print(self.x_train.head())
        print("Train balance")
        listClassCountPercent(self.y_train)
        self.rfr_WithGridSearch = implementRandomForestRegressor.createModelRFRegressorWithGridSearch(self)

    def createModelRFRegressorWithGridSearch(self):
        estimator = RandomForestRegressor(random_state = self.seedRF)
        # scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error']
        modelRFRegressorWithGridSearch = GridSearchCV(estimator, 
                                                    param_grid = self.paramGrid,
                                                    cv = 3,
                                                    n_jobs = -1, 
                                                    verbose = 5, 
                                                    return_train_score = True
                                                    )
        return modelRFRegressorWithGridSearch

    def fitRFRegressorGSearch(self):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        self.rfr_WithGridSearch.fit(self.x_train, y_train)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        bestParameters = self.rfr_WithGridSearch.best_params_
        print(f"The best parameters: {bestParameters}")
        return best_estimator, bestParameters, 
        
    def fitRFRegressorWeighted(self, dominantValeusPenalty):
        y_train= (np.array(self.y_train).astype('float')).ravel()
        weights = ms.createWeightVector(y_train, 0, dominantValeusPenalty)
        self.rfr_WithGridSearch.fit(self.x_train, y_train,sample_weight = weights)
        best_estimator = self.rfr_WithGridSearch.best_estimator_
        bestParameters = self.rfr_WithGridSearch.best_params_
        print(f"The best parameters: {bestParameters}")
        return best_estimator, bestParameters

class ModelTrainCycle():
    def __init__(self,
                 model,
                 modelName,
                 loss_fn,
                 optimizer,
                 labels,
                 pathTrainingDSet,
                 trainingParams, 
                 pathTestDSet = None,
                 scheduler = None, 
                 initWeightfunc= None, 
                 initWeightParams= None, 
                 removeCoordinates = True, 
                 logger:ms.logg_Manager = None,
                 nWorkers = None):
        
        ### Create bases fro training
        self.train_dataset_path = pathTrainingDSet
        self.test_dataset_path = pathTestDSet
        self.model = model
        self.modelName = modelName
        self.logger = logger

        ### Define tracking performance on evaluation Dataset.
        self.trackEval = False ## Default
        self.evaluationLosses = []
        self.evalInBathches = trainingParams.evalInBathches
        if 'evalDatasetPath' in trainingParams.keys():
            self.trackEval = True
            self.evalDatasetPath = trainingParams.evalDatasetPath
            logger.update_logs({'tracking evaluation performance on dataset': self.evalDatasetPath})
            print(f"Self.evalInBathches value == {self.evalInBathches}")

        ## Defining device    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f' The device : {self.device}')
        self.NoCoordinates = removeCoordinates
        self.colsToDrop = trainingParams['colsToDrop']
        self.logger.update_logs({'Cols To Drop': self.colsToDrop})

        ### Load Dataset
        # Read Labels names
        self.logger.update_logs({'Training Set': self.train_dataset_path})
        self.labels = labels
        ## Loading Training Dataset
        self.X, self.Y = self.load_Dataset(self.labels, self.train_dataset_path)

        ## Setting Parameters
        self.epochs = trainingParams['epochs']
        self.logger.update_logs({'Epochs': self.epochs})

        self.splitProportion = trainingParams['testSize']
        self.batchSize = trainingParams['batchSize']
        self.logger.update_logs({'Batch Size': self.batchSize})
        self.saveModelFolder = trainingParams['modelsFolder']
        self.logger.update_logs({'Model saved to ':self.saveModelFolder})
        self.criterion = loss_fn()
        self.initWeightParams = initWeightParams
        self.initWeightFunc = initWeightfunc  
        self.optimizer = optimizer
        self.optimiserInitialState = optimizer
        if nWorkers:
            self.nWorkers = nWorkers
        else: 
            self.nWorkers = 4

        if scheduler is not None:
            self.scheduler = scheduler
            self.schedulerInitialState = scheduler
        
        # Define Models's hidden layer sizes
        input_size = self.X.shape[1]
        self.model = self.model(input_size)
       
        ### Define Optimizer and scheduler
        self.optimizer  = self.optimizer(self.model.parameters())
        if self.scheduler is not None:
            self.scheduler = self.scheduler(self.optimizer)
            self.Sched = True    
        
        ### Init model parameters.
        if self.initWeightFunc  is not None:
                init_params(self.model, self.initWeightFunc, *self.initWeightParams) 
        self.trainLosses = []
        self.valLosses = []

    def splitData_asTensor(self,X,Y)-> tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X.values, Y.values, test_size= self.splitProportion, random_state=42)
        printDataBalace(X_train, X_test, y_train, y_test)
        return torch.tensor(X_train, dtype=torch.float), torch.tensor(X_test, dtype=torch.float), torch.tensor(y_train, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)

    def load_Dataset(self,labels,dataset):
        if self.NoCoordinates:
            X, Y = ms.importDataSet(dataset, labels, self.colsToDrop)
        else: 
            X, Y = ms.importDataSet(dataset, labels)
        return X, Y

    def read_split_as_tensor(self, X_train,y_train,X_test,y_test)-> tuple[torch.tensor,torch.tensor,torch.tensor,torch.tensor]:
        return torch.tensor(X_train.values, dtype=torch.float), torch.tensor(X_test.values, dtype=torch.float), torch.tensor(y_train.values, dtype=torch.float), torch.tensor(y_test.values, dtype=torch.float)
      
    def modelTrainer(self):
        if self.test_dataset_path is None:
            self.X_train, self.X_test, self.y_train, self.y_test = self.splitData_asTensor(self.X,self.Y)
        else:
            X_test, y_test = self.load_Dataset(self.labels,self.test_dataset_path)
            self.X_train, self.X_test, self.y_train, self.y_test = self.read_split_as_tensor(self.X,self.Y,X_test,y_test)
            self.logger.update_logs({'Test Dataset': self.test_dataset_path})
        if self.trackEval:
            X_eval,y_eval = self.load_Dataset(self.labels,self.evalDatasetPath)
            self.X_Eval, self.y_Eval = torch.tensor(X_eval.values, dtype=torch.float),torch.tensor(y_eval.values, dtype=torch.float)
            evaluation_data = TensorDataset(self.X_Eval,self.y_Eval)
            self.evaluation_DataLoader = DataLoader(evaluation_data, batch_size=self.batchSize,shuffle=False, drop_last=False)
        
        # Convert the training set into torch tensors
        self.model, trainMetrics =  self.train(self.X_train, self.X_test, self.y_train, self.y_test)
        modelName = str(self.modelName +'.pkl')
        model_Path = self.saveModelFolder + modelName
        self.logger.update_logs({'model saved as': modelName})
        ms.saveModel(self.model,model_Path)
        return self.model, trainMetrics

    def train(self, X_train,X_test, y_train, y_test)->tuple[nn.Sequential,dict]:
        ### Set loaders
        # Training
        train_data = TensorDataset(X_train, y_train)
        self.train_loader = DataLoader(train_data, batch_size=self.batchSize, 
                                  num_workers = self.nWorkers, shuffle=True)
        # test
        test_data = TensorDataset(X_test, y_test)
        self.test_loader = DataLoader(test_data, batch_size=self.batchSize, 
                                  num_workers = self.nWorkers, shuffle=False, drop_last=False)
        
        # Train the model
        startTime = datetime.now()
        self.model.to(self.device)
        # print('Model deivice',  next(self.model.parameters()).device)
        actual_lr = self.optimizer.param_groups[0]["lr"]
        print(f"Initial Lr : {actual_lr}")
        for e in range(1,self.epochs):
            loopLoss = []
            valLoss = []
            evalLoss = []
            self.model.train()
            for inputs, labels in self.train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                log_probs = self.model(inputs)
                loss = self.criterion(log_probs,labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()
                loopLoss.append(loss.item())
            self.trainLosses.append(np.mean(loopLoss))
            
            ### Compute losses and metrics in the validation Set:
            # Run model on test dataset
            probas,y_hat_test = self.evaluate(X=self.X_test,dataLoader=self.test_loader)
            valLoss = self.criterion(probas,y_test.to(self.device)) # .unsqueeze(1)
            self.valLosses.append(valLoss.item())
                        
            # Run model on Evaluation dataset if required
            if self.trackEval: 
                _,y_hat_Eval = self.evaluate(X=self.X_Eval,dataLoader=self.evaluation_DataLoader)
                evalLoss = self.criterion(y_hat_Eval.to(self.device),self.y_Eval.to(self.device)) # .unsqueeze(1)
                self.evaluationLosses.append(evalLoss.item())
                           
            ### Report every 5 epochs
            if e%5 ==0: 
                ## Log epoch number
                self.logger.update_logs({'<---------Epoch ------->': e })  
                # Compute remining time.
                elapsed_time = datetime.now() - startTime
                avg_time_per_epoch = elapsed_time.total_seconds() / e
                remaining_epochs = self.epochs - e
                estimated_time = remaining_epochs * avg_time_per_epoch
                print(f"Elapsed Time after {e} epochs : {elapsed_time} ->> Estimated Time to End: {ms.seconds_to_datetime(estimated_time)}",' ->actual loss %.4f' %(self.trainLosses[-1]), '-> actual lr = %.8f' %(actual_lr))
                
                ## Report test metric
                probas,y_hat_test = self.evaluate(X=self.X_test,dataLoader=self.test_loader)
                self.logger.update_logs({'Test metric': computeClassificationMetrics(y_test.cpu(),y_hat_test.cpu())})             
                if self.trackEval:
                    ## Report evaluation metrics. 
                    _,y_hat_Eval = self.evaluate(X=self.X_Eval,dataLoader=self.evaluation_DataLoader)
                    self.logger.update_logs({'Evaluation metrics': computeClassificationMetrics(self.y_Eval.cpu(),y_hat_Eval.cpu())}) 
                    
            ### Do schedule step.         
            if self.Sched:   
                self.scheduler.step(loss.item())
                actual_lr = self.optimizer.param_groups[0]["lr"]      

        ###  Log the final metrics
        probas,y_hat_test = self.evaluate(X=self.X_test,dataLoader=self.test_loader) 
        self.logger.update_logs({'Final test metric':computeClassificationMetrics(y_test.cpu(),y_hat_test.cpu())})
        if self.trackEval:
            _,y_hat_Eval = self.evaluate(X=self.X_Eval,dataLoader=self.evaluation_DataLoader)
            self.logger.update_logs({'Final Evaluation metric': computeClassificationMetrics(self.y_Eval.cpu(),y_hat_Eval.cpu())})
        
        return self.model, metrics 

    def evaluate(self,X=None,dataLoader=None, threshold:float=0.5):
        '''
        @X: Features in row format.
        @Y: Labels column vector.
        @dataLoader: Dataloader with the desired dataset to be evaluated in batches. 
        @threshold: Threshold to apply when converting probablilities to binary mask. 
        @Return :  probas,y_hat,metrics (F1-macro-score, F1-micro-score)
        '''
        if self.evalInBathches:
            return self.evaluateModelInBatches(dataLoader,threshold)
        else:
            return self.evaluateModel(X,threshold)

    def evaluateModel(self,X,threshold:float=0.5)-> tuple[np.array,np.array,dict]:
        '''
        @X: Features in row format.
        @Y: Labels column vector.
        @threshold: Threshold to apply when converting probablilities to binary mask. 
        @Return :  probas, y_hat, metrics (F1-macro-score, F1-micro-score)
        '''
        with torch.no_grad():
            self.model.eval()
            inputs = X.to(self.device)
            probas = self.model(inputs)
            y_hat = (probas > threshold).float()
        return probas,y_hat

    def evaluateModelInBatches(self,data_loader,threshold:float=0.5)-> tuple[np.array,np.array,dict]:      
        '''
        @X: Features in row format.
        @Y: Labels column vector.
        @threshold: Threshold to apply when converting probablilities to binary mask. 
        @Return : probas, y_hat, metrics (F1-macro-score, F1-micro-score)
        '''
        with torch.no_grad():
            probs = torch.tensor([]).to(self.device)
            self.model.eval().to(self.device)
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                log_probs = self.model(inputs)
                probs = torch.cat((probs,log_probs.squeeze()))                      
        y_hat = (probs > threshold).float()
        return probs,y_hat

    def plotLosses(self, showIt:bool=True, saveTo:str=''):
        epochs = range(len(self.trainLosses))
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(epochs, self.trainLosses, 'r-', label='Training loss')
        ax.plot(epochs, self.valLosses, 'b--', label='Validation loss')
        # if self.trackEval:
        ax.plot(epochs, self.evaluationLosses, 'k-', label='Evaluation loss')  
        ax.set_title('Training loss vs Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        figName = str(self.modelName +'_losses.png')
        if saveTo:
            plot_Path = saveTo
            self.logger.update_logs({'"--->> Losses vs epochs Figure saved at--->>>':plot_Path})
        else:
            plot_Path = self.saveModelFolder + figName
            self.logger.update_logs({'"--->> Losses vs epochs Figure saved at--->>>':plot_Path})
        
        fig.savefig(plot_Path)
        if showIt:
            fig.show()

    def pltTestConfMatrx(self,showIt:bool=False, saveTo:str=''):
        _,y_hat_test = self.evaluate(X=self.X_test,dataLoader=self.test_loader)
        return plot_ConfussionMatrix(self.y_test,y_hat_test.cpu(),showIt = showIt, saveTo = saveTo)

    

    def plot_ConfussionMatrix(self,y_real,y_hat, showIt:bool=False, saveTo:str=''):
        figName = str(self.modelName +'_ConfMatrx.png')
        
        if saveTo:
            plot_Path = saveTo 
        else:
            plot_Path = self.saveModelFolder + "_" + figName
        plot_ConfussionMatrix(y_real.cpu(),y_hat.cpu(),showIt,plot_Path)

    def getModel(self):
        return self.model
    
    def setModel(self, model):
        self.model = model

    def getLosses(self):
        return self.trainLosses


#### MODELS List
class MLP_1(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.ReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )

    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_2(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )

    def get_weights(self):
            return self.weight
        
    def forward(self,x):
        return self.model(x)

class MLP_3(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size*2),
            nn.LeakyReLU(),
            nn.Linear(input_size*2, input_size*2),
            nn.LeakyReLU(),
            nn.Linear(input_size*2, input_size*2),
            nn.LeakyReLU(),
            nn.Linear(input_size*2, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_4(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )

    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_5(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_5, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size*3),
            nn.LeakyReLU(),
            nn.Linear(input_size*3, input_size*3),
            nn.LeakyReLU(),
            nn.Linear(input_size*3, input_size*3),
            nn.LeakyReLU(),
            nn.Linear(input_size*3, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_6(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_6, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*2),
            nn.LeakyReLU(),
            nn.Linear(input_size*2, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_7(nn.Module):
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_7, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*2),
            nn.LeakyReLU(),
            nn.Linear(input_size*2, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
            nn.Sigmoid(),
        )
    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        return self.model(x)

class MLP_FPSigmoid(nn.Module):
    '''
    Train a parametriced Sigmoid fucntion with input of the form (x*a+b), where x->sequential putput; <a,b> are learnable parameters. 
    The goal is to find out if the model expresivity encrease, compared to standard Sigmoid().
    The model itself is identical to  MLP_6(), with the different sigmoid into the fordward method.
    '''
    def __init__(self, input_size, num_classes:int = 1):
        super(MLP_FPSigmoid, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*5),
            nn.LeakyReLU(),
            nn.Linear(input_size*5, input_size*2),
            nn.LeakyReLU(),
            nn.Linear(input_size*2, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, int(input_size/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_size/2), num_classes),
        )
        self.a = nn.Parameter(torch.tensor([0.001]))
        self.b = nn.Parameter(torch.tensor([0.999]))

    def get_weights(self):
        return self.weight
    
    def forward(self,x):
        # torch.sigmoid(self.model(x)*self.alpha+self.beta)
        return torch.sigmoid(self.b * self.model(x) - self.a)
    
class TransformerModel_1(nn.Module):
    def __init__(self, input_size, num_classes:int=1, num_layers:int=2, num_heads:int=1):
        super(TransformerModel_1, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)
        self.a = nn.Parameter(torch.tensor([0.001]))
        self.b = nn.Parameter(torch.tensor([0.999]))
    
    def forward(self, x):
        outTr = self.transformer_encoder(x)
        prob = torch.sigmoid(self.b * self.fc(outTr)- self.a)
        return prob
    
class TransformerMLPHybrid_1(nn.Module):
    def __init__(self, input_size, num_classes:int=1, num_layers:int=2, num_heads:int=1):
        super(TransformerMLPHybrid_1, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size,num_classes),
        )
        self.a = nn.Parameter(torch.tensor([0.001]))
        self.b = nn.Parameter(torch.tensor([0.999]))
    
    def forward(self, x):
        outTr = self.transformer_encoder(x)
        outMLP = self.mlp(outTr)
        return torch.sigmoid(self.b*outMLP - self.a)

### Helper functions
def split(x,y,TestPercent = 0.2):
    x_train, x_validation, y_train, y_validation = train_test_split( x,y, test_size=TestPercent)
    return x_train, x_validation, y_train, y_validation 

def printDataBalace(x_train, x_validation, y_train, y_validation):
    ## Data shape exploration
    print("",np.shape(x_train),"  :",np.shape(x_validation) )
    unique, counts = np.unique(y_train, return_counts=True)
    print("Label balance on Training set: ", "\n",unique,"\n",counts)
    unique, counts = np.unique(y_validation, return_counts=True)
    print("Label balance on Validation set: ","\n",unique,"\n",counts)

def listClassCountPercent(array):
    '''
      Return the <total> amount of elements and the number of unique values (Classes) 
    in input <array>, with the corresponding count and percent respect to <total>. 
    '''
    unique, count = np.unique(array, return_counts=True)
    total = count.sum()
    result = np.empty_like(unique,dtype='f8')
    result = [(i/total) for i in count]
    listClassCountPercent = {}
    for i in range(len(unique)):
        listClassCountPercent[unique[i]] = str(f"Class_count: {count[i]} (%.4f percent)" %(result[i]))
    return total, listClassCountPercent
 
def createSearshGrid(arg):
    param_grid = {
    'n_estimators': eval(arg['n_estimators']), 
    'max_depth': eval(arg['max_depth']), 
    'max_features': eval(arg['max_features']),
    'max_leaf_nodes': eval(arg['max_leaf_nodes']),
    'bootstrap': eval(arg['bootstrap']),
    }
    return param_grid   

def createSearshGridOneVsRest(arg):
    param_grid = {
    'estimator__n_estimators': eval(arg['estimator__n_estimators']), 
    'estimator__max_depth': eval(arg['estimator__max_depth']), 
    'estimator__max_features': eval(arg['estimator__max_features']),
    'estimator__max_leaf_nodes': eval(arg['estimator__max_leaf_nodes']),
    'n_jobs': eval(arg['n_jobs']), 
    'verbose': eval(arg['verbose']),
    }
    return param_grid   

def quadraticRechapeLabes(x, a, b):
    '''
    Apply quadratic function to a vector like : y = aX^2 + bX.
    You're responsable for providing a and b 
    '''
    x = np.array(x.copy())
    v = (a*x*x) + (b*x) 
    return v.ravel()

def buildInterval(loops,center):
    '''
    Helper function for cyclical searching. The function creates intervals arround some <center>, periodically reducing 
    the step size. 
    @start and @end can be customized into the function. 
    n = number of times the function has been called. 
    center = center of interval
    '''
    start:np.int16
    end:np.int16
    if loops > 1:
        if center >=100:
                start = center-40
                end = center+50
                stepSize = 10  
        else:
            start = center-10
            end = center+11
            stepSize = 2
            if center <= 10 : start = 1
            return np.arange(start,end,stepSize).astype(int)
    else:
            start = center-10
            end = center+11
            stepSize = 2
            if center <= 10 : start = 1  
    return np.arange(start,end,stepSize).astype(int)  

def init_params(model, init_func, *params, **kwargs):
        '''
        Methode to initialize the model's parameters, according a function <init_func>,
            and the necessary arguments <*params, **kwargs>. 
        Change the type of <p> in <if type(p) == torch.nn.Conv2d:> for a different behavior. 
        '''
        for p in model.parameters():
            # if type(p) == (torch.nn.Conv2d or torch.nn.Conv1d) :   # Add a condition as needed. 
            init_func(p, *params)

### Metrics ####
def roc_auc_score_calculation(y_validation, y_hat):
        '''
        Compute one-vs-all for every single class in the dataset
        From: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
        '''
        unique_class = y_validation.unique()
        roc_auc_dict = {}
        if len(unique_class)<=2:
            rocBinary = roc_auc_score(y_validation, y_hat, average = "macro")
            roc_auc_dict['ROC'] = rocBinary
            return roc_auc_dict   
        for per_class in unique_class:
            other_class = [x for x in unique_class if x != per_class]
            new_y_validation = [0 if x in other_class else 1 for x in y_validation]
            new_y_hat = [0 if x in other_class else 1 for x in y_hat]
            roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
            roc_auc_dict[per_class] = roc_auc
        return roc_auc_dict

def plot_ROC_AUC(y_test, y_hat, y_prob):
    '''
    Allows to plot multiclass classification ROC_AUC (One vs Rest), by computing tpr and fpr of each calss respect the rest. 
    The simplest application is the binary classification.
    '''
    unique_class = y_test.unique()
    unique_class.sort()
    print("UNIQUE CLASSES: ", unique_class)
    print("Test Set balance:" )
    listClassCountPercent(y_test)
    print("Prediction balance:")
    listClassCountPercent(y_hat)  
    fig, axs = plt.subplots(1,figsize=(13,4), sharey=True)
    plt.rcParams.update({'font.size': 14})
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.figure(0).clf()
    roc_auc_dict = {}
    i = 0
    if len(unique_class)<=2:  ## Binary case
            fpr,tpr,thresholds = metrics.roc_curve(y_test, y_prob[:,1], drop_intermediate=False)  ### TODO Results doen't match roc_auc_score..
            print(thresholds)
            roc_auc = roc_auc_score(y_test, y_hat, average = "macro")
            axs.plot(fpr,tpr,label = "Class "+ unique_class[1] + " AUC : " + format(roc_auc,".4f")) 
            axs.legend()
    else: 
        for per_class in unique_class:  # Multiclass
            y_testInLoop = y_test.copy()
            other_class = [x for x in unique_class if x != per_class]
            print(f"actual class: {per_class} vs rest {other_class}")
            new_y_validation = [0 if x in other_class else 1 for x in y_testInLoop]
            new_y_hat = [0 if x in other_class else 1 for x in y_hat]
            print(f"Class {per_class} balance vs rest")
            listClassCountPercent(new_y_validation)
            roc_auc = roc_auc_score(new_y_validation, new_y_hat, average = "macro")
            roc_auc_dict[per_class] = roc_auc
            fpr, tpr, _ = metrics.roc_curve(new_y_validation, y_prob[:,i],drop_intermediate=False)  ### TODO Results doen't match roc_auc_score..
            axs.plot(fpr,tpr,label = "Class "+ str(per_class) + " AUC : " + format(roc_auc,".4f")) 
            axs.legend()
            i+=1
    return roc_auc_dict

def accuracyFromConfusionMatrix(confusion_matrix):
    '''
    Only for binary
    '''
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def pritnAccuracy(y_predic, y_val):
    '''
    Only for binary
    '''
    cm = confusion_matrix(y_predic, y_val) 
    print("Accuracy of MLPClassifier : ", accuracyFromConfusionMatrix(cm)) 

def plot_ConfussionMatrix(y_val,y_hat,showIt:bool=True, saveTo:str=''):
    matrix = confusion_matrix(y_val,y_hat)
    print(f"matrix into plotConfusionMAtrix : {matrix}")
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Create the heatmap
    im = ax.imshow(matrix, cmap= 'Blues')
   
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                        matrix.flatten()/np.sum(matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # Add value annotations
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{labels[i, j]}", ha="center", va="center", color="black")
    
   # Customize labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    if saveTo:
        fig.savefig(saveTo)
        print(f"--->> Confusion matrix Saved at --->>>  {saveTo}")
    if showIt:
        plt.show()
    return im

### Evaluation Function:

def evaluateModel(X_DF:pd.DataFrame = None, Y_real_DF:pd.DataFrame = None, model=None, threshold:float=0.5, evalInBathches = True)-> tuple[pd.DataFrame,np.array,np.array]:
    '''
    @X: Features in row format.
    @dataSetPath: dataSetPath to dataset to be evaluated in batches. 
    @threshold: Threshold to apply when converting probablilities to binary mask. 
    @Return : y_real,probas,y_hat
    '''
    X = torch.tensor(X_DF.values, dtype=torch.float)
    Y_real = torch.tensor(Y_real_DF.values, dtype=torch.float) 
    if evalInBathches:
        data = TensorDataset(X, Y_real)
        dataLoader = DataLoader(data, batch_size=10, num_workers = 4, shuffle=False, drop_last=False)
        probas, y_hat = evaluateModel_InBatches(dataLoader,model,threshold)
        return Y_real, probas, y_hat
    else:
        probas, y_hat = evaluateModel_oneShot(X,model,threshold)
        return Y_real,probas,y_hat 

def evaluateModel_oneShot(X,model,threshold)-> tuple[np.array,np.array]:
    '''
    @X: Features in row format.
    @Y: Labels column vector.
    @threshold: Threshold to apply when converting probablilities to binary mask. 
    @Return :  probas, y_hat, metrics (F1-macro-score, F1-micro-score)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval().to(device)
        inputs = X.to(device)
        probas = model(inputs)
        y_hat = (probas > threshold).float()
    return probas,y_hat

def evaluateModel_InBatches(data_loader,model,threshold)-> tuple[np.array,np.array]:      
    '''
    @X: Features in row format.
    @Y: Labels column vector.
    @threshold: Threshold to apply when converting probablilities to binary mask. 
    @Return : probas, y_hat, metrics (F1-macro-score, F1-micro-score)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        probs = torch.tensor([]).to(device)
        model.eval().to(device)
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            log_probs = model(inputs)
            probs = torch.cat((probs,log_probs)) #.squeeze()                     
    y_hat = (probs > threshold).float()
    return probs,y_hat
       
### Inference Function
def inference(model,inferenceDataSet, colsToDrop:list = None,threshold:float=0.5,inferInBatches = False)-> np.array:
        X = pd.read_csv(inferenceDataSet, index_col = None)
        if colsToDrop is not None:
            X.drop(colsToDrop, axis=1, inplace = True)
            print(f'Features for Inference {X.columns}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        X_val = torch.tensor(X.values, dtype=torch.float)
        if inferInBatches:
            infer_Dataset = TensorDataset(X_val)
            # print(infer_Dataset)
            infer_loader = DataLoader(infer_Dataset, batch_size=10,shuffle=False)
            # print(infer_loader)
            probs = torch.tensor([]).to(device)
            model.eval().to(device)
            with torch.no_grad():
                for inputs in infer_loader:
                    inputs_Tsr = torch.tensor(np.array(inputs), dtype=torch.float)
                    inputs_Tsr = inputs_Tsr.to(device)
                    log_probs = model(inputs_Tsr)
                    probs = torch.cat((probs,log_probs.squeeze()))  
            y_hat = (probs > threshold).float().cpu()    
            return y_hat,probs.cpu()
        else:
            with torch.no_grad():
                model.eval().to(device)
                inputs = X_val.to(device)
                probs = model(inputs)
                y_hat = (probs > threshold).float().cpu()
            return y_hat,probs.cpu()

### In test zone. Not ready yet. 
class getEstimatorMetric():
    def __init__(self, cfg:DictConfig):
        local = cfg.local
        pathDataset = local + cfg['pathDataset']
        self.model = local + cfg.estimator
        self.targetColName = cfg.targetColName
        self.x_validation, self.y_validation = ms.importDataSet(self.pathDataset, self.targetColName)
        pass

    def classifierMetric(self):
        classifier = ms.loadModel(self.model)
        y_hat = classifier.predict(self.x_validation)
        accScore = metrics.accuracy_score(self.y_validation, y_hat)
        macro_averaged_f1 = metrics.f1_score(self.y_validation, y_hat, average = 'macro') # Better for multiclass
        micro_averaged_f1 = metrics.f1_score(self.y_validation, y_hat, average = 'micro')
        ROC_AUC_multiClass = roc_auc_score_calculation(self.y_validation,y_hat)
        metric = {}
        metric['accScore'] = accScore
        metric['macro_averaged_f1'] = macro_averaged_f1
        metric['micro_averaged_f1'] = micro_averaged_f1
        metric['ROC_AUC_multiClass'] = ROC_AUC_multiClass
        return metric

    def regressorMetric(self):
        regressor = ms.loadModel(self.model)
        r2_validation = validateWithR2(regressor,self.x_validation,self.y_validation,0,0.1, weighted = False)
        return r2_validation
            
    # Defining a function to decide which function to call
    def metric(self, modelType: str):
        method = self.switch(self,modelType)
        if method != False:
            return method
        else: 
            return print("Invalide model type. Vefify configurations")

    def switch(self, modelType:str):
        dict={
            'regressor': self.regressorMetric(self),
            'classifier': self.classifierMetric(self),
        }
        return dict.get(modelType, False)

def computeMetric(cfg: DictConfig):
    estimatorMetric = getEstimatorMetric(cfg)
    metricResults = estimatorMetric.metric(cfg.modelType)
    print(metricResults)
    return metricResults
 
######################################
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        """
        Calculates the mean absolute error (MAE) between predicted and target tensors.
        Args:
            predicted (torch.Tensor): Predicted values.
            target (torch.Tensor): Ground truth values.
        Returns:
            torch.Tensor: Mean absolute error.
        """
        loss = torch.abs(predicted - target).mean()
        return loss

##### Explore best Sigmoid threshold
##define thresholds

def probs_to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).float()

def exploreBestSigmoidThreshold_F1Macro(y_validation,probs)-> float:
    print('----  Best Macro Average values  -----')
    thresholds = np.arange(0, 1, 0.01)
    macro = [metrics.f1_score(y_validation, probs_to_labels(probs,t), average = 'macro') for t in thresholds]
    ix = np.argmax(macro)
    score = metrics.f1_score(y_validation, probs_to_labels(probs,thresholds[ix]), average = 'micro')
    # print('Threshold=%.3f, F1_Macro=%.4f , score=%.4f' % (thresholds[ix], macro[ix],score))
    return thresholds[ix],macro[ix], score

def exploreBestSigmoidThreshold_F1Score(y_validation,probs):
    thresholds = np.arange(0, 1, 0.01)
    macro = [metrics.f1_score(y_validation, probs_to_labels(probs,t), average = 'macro') for t in thresholds]
    scores = [metrics.f1_score(y_validation, probs_to_labels(probs,t), average = 'micro') for t in thresholds]
    ix = np.argmax(scores)
    # print('Threshold=%.3f, F1 Score=%.5f' % (thresholds[ix], scores[ix]))
    return thresholds[ix],macro[ix],scores[ix]


##
        ###################################################
        ####  METRICS And Feature importance methods  #####
        ###################################################

#### Explore feauture importance PERMUTATION  method  ####
def permutation_importance(datasetPath, labels, model, metric, colsToDrop=None, inBatches:bool = True):
    """
    @Apply to all kind of models.
    
    Calcula la importancia de las características utilizando la técnica de permutación.
    Args:
    - X: numpy array. Conjunto de datos de características.
    - y_true: numpy array. Vector de verdad de terreno.
    - model: objeto de modelo entrenado.
    - metric: función de métrica para evaluar el modelo. Debe tomar dos argumentos: y_true, y_pred.
    Returns:
    - feature_importances: lista de tuplas (nombre de característica, importancia)
    - feature_importance_percentages: lista de tuplas (nombre de característica, porcentaje de importancia)
    """
    X, y_true = ms.importDataSet(datasetPath, labels, colsToDrop)
    # Predicciones con el modelo original
    _,_, y_hat = evaluateModel(X, y_true, model, evalInBathches=inBatches)
   
       #original score
    orig_score = metric(y_true, y_hat.cpu())    
    # Inicializar listas para almacenar importancias y porcentajes
    feature_importances =[]
    feature_importance_percentages = []

    #Calcular importancia de características
    for i in range(0,X.shape[1]):
        # Copiar los datos originales para preservarlos
        X_permuted = X.copy()
        np.random.shuffle(X_permuted.values[:,i])
        # X_permuted_tensor = torch.tensor(X_permuted, dtype=torch.float).to(device)
        # Permutar los valores de una característica
        # Calcular predicciones con la característica permutada
        _,_, y_pred_permuted = evaluateModel(X_permuted, y_true, model, evalInBathches=inBatches)
        # Calcular la métrica con la característica permutada
        permuted_score = metric(y_true, y_pred_permuted.cpu())
        # Calcular la importancia de la característica
        feature_importance = orig_score - permuted_score
        # Agregar la importancia a la lista
        feature_importances.append((i, feature_importance))
    
    # Ordenar características por importancia
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    # Calcular porcentajes de importancia
    total_importance = sum(importance for _, importance in feature_importances)
    for idx, importance in feature_importances:
        feature_importance_percentages.append((idx, (importance / total_importance) * 100))
    
    # Imprime las características ordenadas por importancia y sus porcentajes de importancia
    for idx, importance in feature_importances:
        print(f"Feature {idx}: Importance {importance}")

    for idx, percentage in feature_importance_percentages:
        print(f"Feature {idx}: Importance Percentage {percentage}%")
        
    return feature_importances, feature_importance_percentages

#### Explore feauture importance from random forest models  ####
def investigateFeatureImportance(bestModel, x_train, printed = True)-> pd.DataFrame:
    '''
    @Apply to random forest variants. 
    DEFAULT Path to save: "./models/rwReg/" 
    @input: feature matrix
            bestModel and dateName: created when fitting
            x_train: Dataset to extract features names
            printed option: Default = True
    @return: List of features ordered dessending by importance (pandas DF format). 
    '''
    features = x_train.columns
    featuresInModel= pd.DataFrame({'feature': features,
                   'importance': bestModel.feature_importances_}).\
                    sort_values('importance', ascending = False)
    # clasifierNameExtended = clasifierName + "_featureImportance.csv"     
    # featuresInModel.to_csv(clasifierNameExtended, index = None)
    if printed:
        with pd.option_context('display.max_rows', None,
                       'display.max_columns', None,
                       'display.precision', 4,
                       ):
            print(featuresInModel)
    return featuresInModel

#### Explore feauture importance for torch models  ####
def feature_importance_torch(model, dataloader, device):
    model.eval()
    feature_importances = []

    for i, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        acc = accuracy_score(labels.cpu(), preds.cpu())

        for j in range(inputs.shape[1]):  # iterate over each feature
            inputs_clone = inputs.clone()
            inputs_clone[:, j] = torch.rand_like(inputs_clone[:, j])  # permute feature values
            outputs = model(inputs_clone)
            _, preds = torch.max(outputs, 1)
            acc_permuted = accuracy_score(labels.cpu(), preds.cpu())
            feature_importances.append(acc - acc_permuted)  # importance is decrease in accuracy
    return feature_importances

####  METRICS  #####
def intersection_over_union(y_true, y_pred):
    """
    Calcula la intersección sobre unión (IoU) para evaluar la clasificación binaria.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    intersection = tp
    union = tp + fp + fn
    return intersection / union

def F1_Macro_Average(y_real, y_predicted):
    return metrics.f1_score(y_real, y_predicted, average = 'macro') # Better for multiclass

def F1_Micro_Average(y_real, y_predicted):
    return metrics.f1_score(y_real, y_predicted, average = 'micro') # Better for multiclass

def validateWithR2(model, x_test, y_test, dominantValue:float, dominantValuePenalty:float, weighted = True):
    y_hate = model.predict(x_test)
    if weighted:
        weights = ms.createWeightVector(y_test,dominantValue,dominantValuePenalty)
        r2 = metrics.r2_score(y_test, y_hate,sample_weight = weights)
    else: 
        r2 = metrics.r2_score(y_test, y_hate)
    return r2

def computeClassificationMetrics(y_real,y_predicted)->dict:
    '''
    Ref: https://www.kaggle.com/code/nkitgupta/evaluation-metrics-for-multi-class-classification/notebook
    '''
   #accScore = metrics.accuracy_score(y_real, y_predicted)
    macro_averaged_f1 = F1_Macro_Average(y_real, y_predicted) # Better for multiclassy_real, y_predicted, average = 'macro') # Better for multiclass
    micro_averaged_f1 = F1_Micro_Average(y_real, y_predicted)
    IoU = intersection_over_union(y_real, y_predicted)
    # ROC_AUC_multiClass = roc_auc_score_calculation(y_validation,y_hat)
    # print('Accuraci_score: ', accScore)  
    # print('F1_macroAverage: ', macro_averaged_f1)  
    # print('F1_microAverage: ', micro_averaged_f1)
    # print('ROC_AUC one_vs_all: ', ROC_AUC_multiClass)
    metric = {'macro_averaged_f1' : round(macro_averaged_f1,4),'micro_averaged_f1' : round(micro_averaged_f1,4),'IoU' : round(IoU ,4) }   
    return metric

def computeMainErrors(model, x_test, y_test ):
    y_test  = (np.array(y_test).astype('float')).ravel() 
    y_pred = model.predict(x_test)
    mae = metrics.mean_absolute_error(y_test, y_pred) 
    mse= metrics.mean_squared_error(y_test, y_pred)
    r_mse = np.sqrt(mse) # Root Mean Squared Error
    return mae, mse, r_mse

def reportErrors(model, x_test, y_test):
    '''
    Print the outputs os computeMainError function. 
    '''
    mae, mse, r_mse = computeMainErrors(model, x_test, y_test)
    print('Mean Absolute Error: ',mae )  
    print('Mean Squared Error: ', mse)  
    print('Root Mean Squared Error: ',r_mse)
    


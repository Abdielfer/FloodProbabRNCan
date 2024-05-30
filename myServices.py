import os,ntpath, sys
import glob
import pathlib
import shutil
import joblib
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler 
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from scipy.spatial.distance import cdist


## GIS
from osgeo import gdal,ogr,osr
from osgeo import gdal_array

### Interactive visualization imports
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import Slider, ColumnDataSource, CustomJS
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap, linear_cmap

## Multiprocessing 
import multiprocessing
import concurrent.futures

### Colorlogs. Not necessary, just fun !!!
# import coloredlogs
# coloredlogs.DEFAULT_FIELD_STYLES = {'asctime': {'color': 28},'levelname': {'bold': True, 'color': 'blue'}, 'name': {'color': 'blue'}, 'programname': {'color': 'cyan'}}
# coloredlogs.DEFAULT_LEVEL_STYLES = {'critical': {'bold': True, 'color': 'red'}, 'debug': {'color': 'green'}, 'error': {'color': 'red'}, 'info': {}, 'notice': {'color': 'magenta'}, 'spam': {'color': 'green', 'faint': True}, 'success': {'bold': True, 'color': 'green'}, 'verbose': {'color': 'blue'}, 'warning': {'color': 'yellow'}}
# coloredlogs.COLOREDLOGS_LEVEL_STYLES='spam=22;debug=28;verbose=34;notice=220;warning=202;success=118,bold;error=124;critical=background=red'


### General applications ##
class timeit(): 
    '''
    to compute execution time do:
    with timeit():
         your code, e.g., 
    '''
    def __enter__(self):
        self.tic = datetime.now()
    def __exit__(self, *args, **kwargs):
        print('runtime: {}'.format(datetime.now() - self.tic))

def seconds_to_datetime(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}:{minutes}:{seconds}'

def makeNameByTime():
    name = time.strftime("%y%m%d%H%M")
    return name

def get_parenPath_name_ext(filePath):
    '''
    Ex: user/folther/file.txt
    parentPath = pathlib.PurePath('/src/goo/scripts/main.py').parent 
    parentPath => '/src/goo/scripts/'
    parentPath: can be instantiated.
         ex: parentPath[0] => '/src/goo/scripts/'; parentPath[1] => '/src/goo/', etc...
    '''
    parentPath = pathlib.PurePath(filePath).parent
    fpath = pathlib.Path(filePath)
    ext = fpath.suffix
    name = fpath.stem
    return parentPath, name, ext
  
def addSubstringToName(path, subStr: str, destinyPath = None) -> os.path:
    '''
    @path: Path to the raster to read. 
    @subStr:  String o add at the end of the origial name
    @destinyPath (default = None)
    '''
    parentPath,name,ext= get_parenPath_name_ext(path)
    if destinyPath != None: 
        return os.path.join(destinyPath,(name+subStr+ext))
    else: 
        return os.path.join(parentPath,(name+subStr+ ext))

def createListFromCSV(csv_file_location: os.path, delim:str =',')->list:  
    '''
    @return: list from a <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    '''      
    print('List created from csv : ->>', csv_file_location) 
    df = pd.read_csv(csv_file_location, index_col= None, header=None, delimiter=delim)
    out = []
    for i in range(0,df.shape[0]):
        out.append(df.iloc[i][0])
    return out

def createCSVFromList(pathToSave: os.path, listData:list):
    '''
    This function create a *.csv file with one line per <lstData> element. 
    @pathToSave: path of *.csv file to be writed with name and extention.
    @listData: list to be writed. 
    '''
    with open(pathToSave, 'w') as output:
        for line in listData:
            output.write(str(line) + '\n')
    return True

def abort(msg:str=''):
    sys.exit(msg)

### Hydra
def getHydraJobName():
    return HydraConfig.get().job.name

def getHydra_Runtime_OutDir():
    return hydra.core.hydra_config.HydraConfig.get()['runtime']['output_dir']  

def updateHydraDicConfig(cfg:DictConfig, args:iter):
    outDir = cfg.copy()
    for key, value in args.items():
        print(f"Updating Key: {key}, and value: {value}")
        outDir[key] = value
    return outDir

##### Loggings-
class logg_Manager():
    '''
    This class creates a logger object that writes logs to both a file and the console. 
    @log_name: lLog_name. Logged at the info level by default.
    @log_dict: Dictionary, Set the <attributes> with <values> in the dictionary. 
    The logger can be customized by modifying the logger.setLevel and formatter attributes.

    The update_logs method takes a dictionary as input and updates the attributes of the class to the values in the dictionary. The method also takes an optional level argument that determines the severity level of the log message. 
    '''
    def __init__(self,logName:str=None, log_dict:dict=None):# log_name, 
        super(logg_Manager, self).__init__()
        self.HydraOutputDir = getHydra_Runtime_OutDir()
        print(self.HydraOutputDir)
        self.logger = logging.getLogger(logName)
        # coloredlogs.install(logger=self.logger)
        # coloredlogs.install(fmt='%(asctime)s,%(levelname)s %(message)s')
        ### Fill the logger at creation with a dictionary.
        if log_dict is not None:
            for key, value in log_dict.items():
                setattr(self, key, value)
                self.logger.info(f'{key} - {value}')
        
    def update_logs(self, log_dict, level=None):
        for key, value in log_dict.items():
            setattr(self, key, value)
            if level == logging.DEBUG:
                self.logger.debug(f'{key} - {value}')
            elif level == logging.WARNING:
                self.logger.warning(f'{key} - {value}')
            elif level == logging.ERROR:
                self.logger.error(f'{key} - {value}')
            else:
                self.logger.info(f'{key} - {value}')
    
    # def saveLogsAsTxt(self):
    #     '''
    #     Save the hydra default logger as txt file. 
    #     '''
    #     # Create a file handler
    #     handler = logging.FileHandler(self.HydraOutputDir + self.logger.name + '.txt')
    #     # Set the logging format
    #     formatter = logging.Formatter('%(name)s - %(message)s')
    #     handler.setFormatter(formatter)
    #     # Add the file handler to the logger
    #     self.logger.addHandler(handler)

def reportDatasetDescrition(dataset:os.path,logManager:logg_Manager, excellOutPath:os.path=None)->None:
    '''
    Report dataset descrition:  as excel fiel if <excellOutPath> is not None, otherwise is logged. 
    @return: None
    '''   
    inferDataset = pd.read_csv(dataset,index_col=None)
    descritor = inferDataset.describe()
    if excellOutPath is not None:
        descritor.to_excel(excellOutPath)
    else:
        logManager.update_logs({'Dataset description': str(descritor)}) 
    
    return None

## models manipulation
def saveModel(estimator, name):
    _ = joblib.dump(estimator, name, compress=9)

def loadModel(modelName:os.path):
    return joblib.load(modelName)

def logTransformation(x):
    '''
    Logarithmic transformation to redistribute values between 0 and 1. 
    '''
    x_nonZeros = np.where(x <= 0.0000001, 0.0001, x)
    return np.max(np.log(x_nonZeros)**2) - np.log(x_nonZeros)**2

def createWeightVector(y_vector, dominantValue:float, dominantValuePenalty:float):
    '''
    Create wight vector for sampling weighted training.
    The goal is to penalize the dominant class. 
    This is important is the flood study, where majority of points (usually more than 95%) 
    are not flooded areas. 
    '''
    y_ravel  = (np.array(y_vector).astype('int')).ravel()
    weightVec = np.ones_like(y_ravel).astype(float)
    weightVec = [dominantValuePenalty if y_ravel[j] == dominantValue else 1 for j in range(len(y_ravel))]
    return weightVec

def count_parameters(model)-> tuple[dict,float]:
    table = {}
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table[name]=params
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return table, total_params

####  Sampling manipulation
def importDataSet(csvPath, targetCol: str, colsToDrop:list=None)->tuple[pd.DataFrame,pd.DataFrame]:
    '''
    Import datasets and return         
    @csvPath: DataSetName => The dataset path as *csv file. 
    @Output: Features(x) and tragets(y) 
    ''' 
    x  = pd.read_csv(csvPath, index_col = None)
    y = x[targetCol]
    x.drop([targetCol], axis=1, inplace = True)
    if colsToDrop is not None:
        x.drop(colsToDrop, axis=1, inplace = True)
        print(f'Features in dataset {x.columns}')
    return x, y

def concatDatasets(datsetPathList_csv, outPath:os.path = None):
    outDataSet = pd.DataFrame()
    for file in datsetPathList_csv:
        df = pd.read_csv(file,index_col = None)
        outDataSet = pd.concat([outDataSet,df], ignore_index=True)
    if outPath is not None:
        outDataSet.to_csv(outPath, index=None)
    return outDataSet

def randomUndersampling(x_DataSet, y_DataSet, sampling_strategy = 'auto'):
    sm = RandomUnderSampler(random_state=50, sampling_strategy=sampling_strategy)
    x_res, y_res = sm.fit_resample(x_DataSet, y_DataSet)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res

def stratifiedSplit_FromDataframePath(dataSetName:os.path, targetColName:str='Labels', test_size:float=0.2)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    Performe a sampling that preserve classses proportions on both, train and test sets.
    
    @return: tupel[pd.DataFrame,..,pd.DataFrame]=> [X_train, y_train, X_test, y_test]
    '''
    X,Y = importDataSet(dataSetName, targetColName)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=50)
    for train_index, test_index in sss.split(X, Y):
        print("TRAIN:", train_index.size, "TEST:", test_index)
        X_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
    return X_train, y_train, X_test, y_test

def stratifiedSplit(X,Y,test_size:float=0.2)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    Performe a sampling that preserve classses proportions on both, train and test sets.
    @X: Features dataframe.
    @Y: Labels dataframe column.
    @return: tupel[pd.DataFrame,..,pd.DataFrame]=> [X_train, y_train, X_test, y_test]
    '''
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=50)
    for train_index, test_index in sss.split(X, Y):
        print("TRAIN:", train_index.size, "TEST:", test_index.size)
        X_train = X.iloc[train_index]
        y_train = Y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = Y.iloc[test_index]
    
    return X_train, y_train, X_test, y_test

def removeCoordinatesFromDataSet(dataSet):
    '''
    Remove colums of coordinates if they exist in dataset
    @input:
      @dataSet: pandas dataSet
    '''
    DSNoCoord = dataSet
    if 'x_coord' in DSNoCoord.keys(): 
      DSNoCoord.drop(['x_coord','y_coord'], axis=1, inplace = True)
    else: 
      print("DataSet has no coordinates to remove")
    return DSNoCoord

def transformDatasets(cfg):
    '''
    This fucntion takes a list of dataset as *csv paths and apply a transformation in loop. You can custom your transformation to cover your needs. 
    @cfg:DictConfig: The input must be a Hydra configuration lile, containing the key = datasetsList to a csv with the addres of all *.cvs file to process.  
    '''
    DatasetsPath = cfg.datasetsList
    listOfDatsets = createListFromCSV(DatasetsPath)
    for file in listOfDatsets:
        X,Y = importDataSet(file, targetCol='Labels')
        x_dsUndSamp, Y_dsUndSamp = randomUndersampling(X,Y)
        x_dsUndSamp['Labels'] = Y_dsUndSamp.values
        NewFile = addSubstringToName(file,'_balanced')
        x_dsUndSamp.to_csv(NewFile, index=None)
    pass

def DFOperation_removeNegative(DF:pd.DataFrame,colName):
    '''
    NOTE: OPPERATIONS ARE MADE IN PLACE!!!
    Remove all row index in the collumn <colName> if the value is negative
    '''
    DF = DF[DF.colName>=0]
    return DF

    ###  Dataset similarity measurements.

def datasetsEuclideanDistances(datasetA:pd.DataFrame, datasetB:pd.DataFrame, collsToCompare:list) -> float:
    '''
    Compute de mean euclidean distance between all the samples of two datasets.
    @datasetA:pd.DataFrame: Samples set A.
    @datasetB:pd.DataFrame: Samples set B.
    @collsToCompare:list = []: List of features to be considered in the comparison. If Empty, the full set of columns will be considered for coparison. 
    @Return: Mean distance between all pairs from dataset A and B 
    '''
    startTime = datetime.now()

    if not collsToCompare:
        grup_A = datasetA
        grup_B = datasetB
    else:
        grup_A = datasetA[collsToCompare]
        grup_B = datasetB[collsToCompare]

    maxrow = grup_B.shape[0]
    distancesFromBiToA = np.zeros([maxrow,1])
    
    valuesA = grup_A.values
    valuesB = grup_B.values

    for i in range(maxrow):
        # Compute the Euclidean distance between the tow goups
        value = valuesB[i]
        value = np.reshape(value,(1, -1))
        dist = cdist(valuesA, value,'euclidean')
        distancesFromBiToA[i]=np.mean(dist)
        if i%10000==0:
            elapsed_time = datetime.now() - startTime
            avg_time_per_epoch = elapsed_time.total_seconds()/(i+0.01)
            remaining_epochs = maxrow-i
            estimated_time = remaining_epochs * avg_time_per_epoch
            print(f"Elapsed Time : {elapsed_time} ->> Estimated Time to End: {seconds_to_datetime(estimated_time)}")
            print(f"last distance {np.mean(dist)}")
            print(f"Procesed {i} / {maxrow}")
    
    print(f"Last i : {i}")
    # Compute the median form the euclidean distance. 
    mean_dist = np.mean(distancesFromBiToA, axis=0)

    print("mean euclidean distance between the two datasets:", mean_dist)
    return distancesFromBiToA, mean_dist

def datasetsCosineSimilarity(datasetA:pd.DataFrame, datasetB:pd.DataFrame, collsToCompare:list) -> float:
    '''
    Compute de mean euclidean distance between all the samples of two datasets.
    @datasetA:pd.DataFrame: Samples set A.
    @datasetB:pd.DataFrame: Samples set B.
    @collsToCompare:list = []: List of features to be considered in the comparison. If Empty, the full set of columns will be considered for coparison. 
    @Return: Mean distance between all pairs from dataset A and B 
    '''
    startTime = datetime.now()

    if not collsToCompare:
        grup_A = datasetA
        grup_B = datasetB
    else:
        grup_A = datasetA[collsToCompare]
        grup_B = datasetB[collsToCompare]

    maxrow = grup_B.shape[0]
    distancesFromBiToA = np.zeros([maxrow,1])
    
    valuesA = grup_A.values
    valuesB = grup_B.values

    for i in range(maxrow):
        # Compute the Euclidean distance between the tow goups
        value = valuesB[i]
        value = np.reshape(value,(1, -1))
        dist = cdist(valuesA, value,'cosine')
        distancesFromBiToA[i]=np.mean(dist)
        if i%1000==0:
            elapsed_time = datetime.now() - startTime
            avg_time_per_epoch = elapsed_time.total_seconds()/(i+0.01)
            remaining_epochs = maxrow-1
            estimated_time = remaining_epochs * avg_time_per_epoch
            print(f"Elapsed Time : {elapsed_time} ->> Estimated Time to End: {seconds_to_datetime(estimated_time)}")
            print(f"last distance {np.mean(dist)}")
            print(f"Procesed {i} / {maxrow}")
    
    print(f"Last i : {i}")
    # Compute the median form the euclidean distance. 
    mean_dist = np.mean(distancesFromBiToA, axis=0)

    print("mean euclidean distance between the two datasets:", mean_dist)
    return distancesFromBiToA, mean_dist

### Testing Faiss
def faissDistanceCalculation(datasetA:pd.DataFrame, datasetB:pd.DataFrame, collsToCompare:list):
    
    
    pass
    
### Dataset Pretreatment and visualization
def readDataFrame_FromDiferentSources(datasetSource)->pd.DataFrame:
    '''
    Wrap fuction to read dataframes from path or pandas dataframes, without taking care of the source type. 
    '''
    if isinstance(datasetSource, pd.DataFrame):
        return datasetSource
    
    ## Read dataset source
    if os.path.exists(datasetSource):
        Source = pd.read_csv(datasetSource, index_col=None)
        return Source
    
    else:
        print('Datset Source is nider a path or pd.DataFrame')
        return None

def standardizeDataSetCol(dataSet, colName:str):
    '''
    Perform satandardizartion on a column of a DataFrame
    '''
    dataSet = readDataFrame_FromDiferentSources(dataSet)
    mean = dataSet[colName].mean()
    std = dataSet[colName].std()
    column_estandar = (dataSet[colName] - mean) / std
    return column_estandar,mean,std

def computeStandardizer_fromDataSetCol(dataSet,colName)->tuple[float,float,float,float]:
    '''
    Perform satandardizartion on a column of a DataFrame
    '''
    dataSet = readDataFrame_FromDiferentSources(dataSet)
    min = dataSet[colName].min()
    max = dataSet[colName].max()
    mean = dataSet[colName].mean()
    std = dataSet[colName].std()
    del dataSet
    return mean,std,min,max

def standardizeDatasetByColName(datasetSource:os.path, datasetObjectivePath:os.path=None, colNamesList:list =[])->pd.DataFrame:
    '''
    Standardize a list of columns in a dataset in <datasetPath>, from the values obtained from the <datasetSource>. Both Datasets MUST have the same col_name. 
    (ex. to perform standardization in validation set from the values in the training set.)
    @datasetSource: path to the main dataset from which extract mean and std at each col from <colNamesList>
    @datasetPath: path to the dataset to be standar
    @colNamesList: 
    '''
    ## Read dataset source
    dataSetSource = pd.read_csv(datasetSource, index_col=None)

    ## Read dataset Objective
    dataSetObjective = pd.read_csv(datasetObjectivePath, index_col=None)
    outputDataset = dataSetObjective.copy()

    ## Standardize one col at the time.
    for col in colNamesList:
        mean,std,_,_ = computeStandardizer_fromDataSetCol(dataSetSource, col)
        column_estandar = (dataSetObjective[col] - mean) / std
        outputDataset[col] = column_estandar
    
    del dataSetObjective, dataSetSource
    return outputDataset

def normalizeDatasetByColNameList(datasetSource, datasetObjective=None, colNamesList:list =[])->pd.DataFrame:
    '''
    Standardize a list of columns from a dataset in <datasetObjective>, from the values obtained from the <datasetSource>. Both Datasets MUST have the same col_names. 
    (ex. to perform standardization in validation set from the values in the training set)
    @datasetSource: (path OR pd.DataFrame) to the main dataset from which extract [mim,max] from each column in <colNamesList>.
    @datasetObjective: (path OR pd.DataFrame) to the dataset to be standardized.
    @colNamesList: List of column names, for the clumn to be normalized. Verify Key value error befor call the function. 
    '''

    if len(colNamesList)==0:
        print("NO Column to normalize. Define <colNamesList>")
        return None

    ## Read dataset source
    source = readDataFrame_FromDiferentSources(datasetSource)

    ## Read or assigne dataset objective
    if datasetObjective is None:
        objective = source
    else:
        objective =  readDataFrame_FromDiferentSources(datasetObjective)

    outputDataset = objective.copy()

    ## Compute min-max normalization by column    
    for col in colNamesList:
        _,_,min,max = computeStandardizer_fromDataSetCol(source, col)
        Delta = max-min
        column_estandar = (objective[col] - min) / Delta
        outputDataset[col] = column_estandar
    
    ## Clean memory
    del source, objective
    return outputDataset

def extractFloodClassForMLP(csvPath):
    '''
    THis function asume the last colum of the dataset are the lables
    
    The goal is to create separated Datasets with classes 1 and 5 from the input csv. 
    The considered rule is: 
        Class_5: all class 5.
        Class_1: All classes, since they are inclusive. All class 5 are also class 5. 
    '''
    df = pd.read_csv(csvPath,index_col = None)
    print(df.head())
    labelsName = df.columns[-1]
    labels= df.iloc[:,-1]
    uniqueClasses = pd.unique(labels)
    if 1 in uniqueClasses:
        class1_Col = [1 if i!= 0 else 0 for i in labels]
        df[labelsName] = class1_Col
        dfOutputC1 = addSubstringToName(csvPath,'_Class1')
        df.to_csv(dfOutputC1,index=None)

    if 5 in uniqueClasses:
        class5_Col = [1 if i == 5 else 0 for i in labels]
        df[labelsName] = class5_Col
        dfOutputC5 = addSubstringToName(csvPath,'_Class5')
        df.to_csv(dfOutputC5,index=None)
    
    return dfOutputC1,dfOutputC5

def plotHistComparison(vectorA,vectorB, bins:int = 200):
    x_lim = max(np.max(vectorA), np.max(vectorB))
    # Setting plot
    fig, ax = plt.subplots(1,sharey=True, tight_layout=True)
    ax.hist(vectorA, bins, density=False, histtype='step', label=['VectorA'],stacked=True, fill=False)
    ax.hist(vectorB, bins, density=False, histtype='step', label='VectorB',stacked=True, fill=False)
    ax.legend(prop={'size': 10})
    # ax.set_title('cdem_16m vs srdem_8m') 
    ax.set_xlim([-0.1,x_lim])
    fig.tight_layout()
    plt.show()
    pass

def plotVectorsPDFComparison(vectorList:list,title:str='PDF', bins:int = 100, addmax= False, show:bool=False, save:bool=False, savePath:str='', plotGlobal:bool=False):
    '''
    # this create the kernel, given an array, it will estimate the probability over that values
    kde = gaussian_kde( data )
    # these are the values over which your kernel will be evaluated
    dist_space = linspace( min(data), max(data), 100 )
    # plot the results
    plt.plot( dist_space, kde(dist_space))
    @vectorList:list,
    @title:str='RasterPDF' 
    @bins:int = 100 
    @addmax= False
    @show:bool=False 
    @globalMax:int = 0 
    @save:bool=True 
    @savePath:str=''
    ''' 
    global_Min = 0
    global_Max = -np.inf
    nameList = []
    fullDataSet = np.array([], dtype=np.float32)
    
    ### Prepare plot
    fig, ax = plt.subplots(1,sharey=True, sharex=True, tight_layout=True)
    colors = plt.cm.jet(np.linspace(0, 1, 2*len(vectorList)+10))# 
    '''
    turbo; jet; nipy_spectral;gist_ncar;gist_rainbow;rainbow, brg
    '''
    colorIdx = 0
    for item in vectorList:
        dataMin =  np.min(item)
        relativeElevation = np.subtract(item,dataMin)
        fullDataSet = np.concatenate((fullDataSet,relativeElevation))
        print(f'FullDataset shape {fullDataSet.shape}')
        print(f'data shape {relativeElevation.shape}')
        dataMin =  np.min(relativeElevation)
        dataMax = np.max(relativeElevation)
        global_Min = np.minimum(global_Min,dataMin)
        global_Max = np.maximum(global_Max,dataMax)

        ## If <bins> is a list, add the maximum value to <bins>.  
        if (addmax and isinstance(bins,list)):
            bins.append(global_Max).astype(int)
        counts, bins = np.histogram(relativeElevation, bins= 200)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        ax.plot(bin_centers, counts,color=colors[colorIdx],linewidth=0.8)
        colorIdx+=5
        
        
    # ## Plot global PDF
    counts, bins = np.histogram(fullDataSet, bins= 400)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(bin_centers, counts,color='red',linewidth=0.9, label='global')
    nameList.append('Global histogram')
    
    ## Complet plot space
    ax.legend(nameList, prop={'size': 5})
    ax.set_title(title)
    ax.set_ylabel('Frequency')
   
    if isinstance(bins,list):
        plt.xticks(bins)
        print(bins)
        plt.gca().set_xticklabels([str(i) for i in bins], minor = True)
          
    if save:
        if not savePath:
            savePath = os.path.join(os.getcwd(),title +'.png')
            print(f'Figure: {title} saved to dir: {savePath}')
        plt.savefig(savePath)
    
    if show:
        plt.show()


### Modifying class domain
def pseudoClassCreation(dataset, conditionVariable, threshold, pseudoClass, targetColumnName):
    '''
    Replace <targetClass> by  <pseudoClass> where <conditionVariable >= threshold>. 
    Return:
      dataset with new classes group. 
    '''
    datasetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetColumnName])).ravel()
    conditionVar = (np.array(dataset[conditionVariable])).ravel()
    datasetReclassified[targetColumnName] = [ pseudoClass if conditionVar[j] >= threshold 
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datasetReclassified[targetColumnName]))
    return  datasetReclassified

def revertPseudoClassCreation(dataset, originalClass, pseudoClass, targetColumnName):
    '''
    Restablich  <targetClass> with <originalClass> where <targetColumnName == pseudoClass>. 
    Return:
      dataset with original classes group. 
    '''
    datasetReclassified = dataset.copy()
    actualTarget = (np.array(dataset[targetColumnName])).ravel()
    datasetReclassified[targetColumnName] = [ originalClass if actualTarget[j] == pseudoClass
                                           else actualTarget[j]
                                           for j in range(len(actualTarget))]
    print(Counter(datasetReclassified[targetColumnName]))
    return  datasetReclassified

def makeBinary(dataset,targetColumn,classToKeep:int, replacerClassName:int):
    '''
    makeBinary(dataset,targetColumn,classToKeep, replacerClassName)
    @classToKeep @input: Class name to conserv. All different classes will be repleced by <replacerClassName>
    '''
    repalcer  = dataset[targetColumn].to_numpy()
    dataset[targetColumn] = [replacerClassName if repalcer[j] != classToKeep else repalcer[j] for j in range(len(repalcer))]  
    return dataset


### Configurations And file management
def importConfig():
    with open('./config.txt') as f:
        content = f.readlines()
    print(content)    
    return content

def getLocalPath():
    return os.getcwd()

def makePath(str1,str2):
    return os.path.join(str1,str2)

def ensureDirectory(pathToCheck):
    if os.path.isdir(pathToCheck): 
        return pathToCheck
    else:
        os.mkdir(pathToCheck)
        print(f"Confirmed directory at: {pathToCheck} ")
        return pathToCheck

def relocateFile(inputFilePath, outputFilePath):
    '''
    NOTE: @outputFilePath ust contain the complete filename
    Sintax:
     @shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    '''
    shutil.move(inputFilePath, outputFilePath)
    return True

def createTransitFolder(parent_dir_path, name:str='TransitDir'):
    path = os.path.join(parent_dir_path, name)
    ensureDirectory(path)
    return path

def clearTransitFolderContent(path, filetype = '/*'):
    '''
    NOTE: This well clear dir without removing the parent dir itself. 
    We can replace '*' for an specific condition ei. '.tif' for specific fileType deletion if needed. 
    @Arguments:
    @path: Parent directory path
    @filetype: file type toi delete. @default ='/*' delete all files. 
    '''
    files = glob.glob(path + filetype)
    for f in files:
        os.remove(f)
    return True

def listFreeFilesInDirByExt(cwd:str, ext = '.tif'):
    '''
    @ext = *.tif by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root, dirs, file) in os.walk(cwd):
        for f in file:
            # print(f"File: {f}")
            _,_,extent = get_parenPath_name_ext(f)
            if extent == ext:
                file_list.append(f)
    return file_list

def listFreeFilesInDirByExt_fullPath(cwd:str, ext = '.csv') -> list:
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root,_, file) in os.walk(cwd, followlinks=True):
        for f in file:
            # print(f"Current f: {f}")
            _,extent = splitFilenameAndExtention(f)
            # print(f"Current extent: {extent}")
            if ext == extent:
                file_list.append(os.path.join(root,f))
    return file_list

def listFreeFilesInDirBySubstring_fullPath(cwd:str, substring = '') -> list:
    '''
    @substring: substring to be verify onto the file name. 
    NOTE:  THIS function list only files that are directly into <cwd> path. 
    '''
    cwd = os.path.abspath(cwd)
    # print(f"Current working directory: {cwd}")
    file_list = []
    for (root,_, file) in os.walk(cwd, followlinks=True):
        for f in file:
            if substring.lower() in f.lower():
                file_list.append(os.path.join(root,f))
    return file_list

def listALLFilesInDirByExt(cwd, ext = '.csv'):
    '''
    @ext = *.csv by default.
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList: list = []
    for (root, _, _) in os.walk(cwd):
         fullList.extend(listFreeFilesInDirByExt(root, ext)) 
    return fullList

def listALLFilesInDirByExt_fullPath(cwd, ext = '.csv'):
    '''
    @ext: NOTE <ext> must contain the "." ex: '.csv'; '.tif'; etc...
    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList = []
    for (root, _, _) in os.walk(cwd):
        # print(f"Roots {root}")
        localList = listFreeFilesInDirByExt_fullPath(root, ext)
        # print(f"Local List len :-->> {len(localList)}")
        fullList.extend(localList) 
        return fullList

def listALLFilesInDirBySubstring_fullPath(cwd, substring = '.csv')->list:
    '''
    @substring: substring to be verify onto the file name.    NOTE:  THIS function list ALL files that are directly into <cwd> path and children folders. 
    '''
    fullList = []
    for (root, _, _) in os.walk(cwd):
        # print(f"Roots {root}")
        localList = listFreeFilesInDirBySubstring_fullPath(root, substring)
        print(f"Local List len :-->> {len(localList)}")
        fullList.extend(localList) 
        return fullList

def createListFromCSVColumn(csv_file_path, col_id:str):  
    '''
    @return: list from <col_id> in <csv_file_location>.
    Argument:
    @csv_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.read_csv(csv_file_path, index_col = None)
    for i in df[col_id]:
        x.append(i)
    return x

def createListFromExcelColumn(excell_file_location,Sheet_id:str, col_id:str):  
    '''
    @return: list from <col_id> in <excell_file_location>.
    Argument:
    @excell_file_location: full path file location and name.
    @col_id : number of the desired collumn to extrac info from (Consider index 0 for the first column)
    '''       
    x=[]
    df = pd.ExcelFile(excell_file_location).parse(Sheet_id)
    for i in df[col_id]:
        x.append(i)
    return x

def splitFilenameAndExtention(file_path):
    '''
    pathlib.Path Options: 
    '''
    fpath = pathlib.Path(file_path)
    extention = fpath.suffix
    name = fpath.stem
    return name, extention 

def replaceExtention(inPath,newExt: str)->os.path :
    '''
    Just remember to add the poin to the new ext -> '.map'
    '''
    dir,fileName = ntpath.split(inPath)
    _,actualExt = ntpath.splitext(fileName)
    return os.path.join(dir,ntpath.basename(inPath).replace(actualExt,newExt))

def writeReportFromDict(info:dict,outPath:os.path):
    metricToSave = pd.DataFrame.from_dict(info, orient='index')
    metricToSave.to_csv(outPath,index = True, header=True)

###########            
### GIS ###
###########

def readRasterAsArry(rasterPath):
   return gdal_array.LoadFile(rasterPath)

def setInformationToCreateShpFile(datasetPath,y_hat,targetCol)->os.path:
    '''
    We asume here that <datasetPath> contain coordinates as <x_coord> and <y_coord>.
    Return:
         The path of the *csv, with the new DtataFrame, containing four columns ['x_coord','y_coord',targetCol,'prediction'].  
    '''
    dataFrame  = pd.read_csv(datasetPath, index_col = None)
    ds_toSHP = pd.DataFrame()
    if 'Labels' in dataFrame.columns:
        ds_toSHP['x_coord'] = dataFrame['x_coord'].astype('Float32')
        ds_toSHP['y_coord'] = dataFrame['y_coord'].astype('Float32')
        ds_toSHP['Labels'] = dataFrame[[targetCol]]
    else:
        ds_toSHP['x_coord'] = dataFrame['x_coord'].astype('Float32')
        ds_toSHP['y_coord'] = dataFrame['y_coord'].astype('Float32')
          
    ds_toSHP['prediction'] = y_hat
    outDataFrameCsv = addSubstringToName(datasetPath,"_4Shp")
    ds_toSHP.to_csv(outDataFrameCsv,index=False)
    print(outDataFrameCsv)
    return outDataFrameCsv

def buildShapefilePointFromCsvDataframe(csvDataframe:os.path, outShepfile:os.path='', EPGS:int=3979)->os.path:
    '''
    Creates a shapefile of points from a Dataframe <df>. The DataFrame is expected to have a HEADER, and the two first colums with the x_coordinates and y_coordinates respactivelly.
    @csvDatafame:os.path: Path to the *csv containing the Dataframe with the list of points to add to the Shapefile. 
    @outShepfile:os.path: Output path to the shapefile (Optional). If Non, the shapefile will have same path and name that csvDataframe.
    @EPGS: EPGS value of a valid reference system (Optional).(Default = 4326).
    
    @Return: Path to out shapefile.
    '''
    df = pd.read_csv(csvDataframe)
    #### Create a new shapefile
    ## Set shapefile path.
    if outShepfile:
        outShp = outShepfile
    else: 
        outShp = replaceExtention(csvDataframe,'.shp')
        print(outShp)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.CreateDataSource(outShp)

    # Set the spatial reference
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPGS)  # WGS84

    # Create a new layer
    layer = ds.CreateLayer("", srs, ogr.wkbPoint)

    # Add fields
    for column in df.columns:
        field = ogr.FieldDefn(column, ogr.OFTReal)
        field.SetWidth(10)  # Total field width
        field.SetPrecision(2)  # Width of decimal part
        layer.CreateField(field)

    # Add points
    for _,row in df.iterrows():
        # Create a new feature
        feature = ogr.Feature(layer.GetLayerDefn())
        # Set the attributes
        for column in df.columns:
            feature.SetField(column, row[column])
        # Create a new point geometry
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(row[0], row[1])
        # Set the feature geometry
        feature.SetGeometry(point)
        # Create the feature in the layer
        layer.CreateFeature(feature)
        # Dereference the feature
        feature = None

    # Dereference the data source
    ds = None
    return outShp

def rasterizePointsVector(vectorInput, rasterOutput, attribute:str='fid',pixel_size:int = 1,EPGS:str = 'EPSG:3979')->bool:
    # Open the data source
    ds = ogr.Open(vectorInput)
    lyr = ds.GetLayer()
    # Set up the new raster
    # this should be set appropriately
    x_min, x_max, y_min, y_max = lyr.GetExtent()
    cols = int((x_max - x_min) / pixel_size)
    rows = int((y_max - y_min) / pixel_size)
    raster_ds = gdal.GetDriverByName('GTiff').Create(rasterOutput, cols, rows, 1, gdal.GDT_Float32)
    raster_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    raster_ds.SetProjection(EPGS)
    band = raster_ds.GetRasterBand(1)
    band.SetNoDataValue(0)

    # Rasterize
    gdal.RasterizeLayer(raster_ds, [1], lyr, options=[f"ATTRIBUTE={attribute}",'COMPRESS=LZW','TILED=YES','BIGTIFF=YES'])

    # Close datasets
    band = None
    raster_ds = None
    ds = None
    return True

def pointsVectorToNPArray(vectorInput,pixel_size:int = 1,EPGS:str = 'EPSG:3979')->np.array:
    # Open the data source
    ds = ogr.Open(vectorInput)
    lyr = ds.GetLayer()
    # Set up the new raster
    # this should be set appropriately
    x_min, x_max, y_min, y_max = lyr.GetExtent()
    cols = int((x_max - x_min) / pixel_size)
    rows = int((y_max - y_min) / pixel_size)
    raster_ds = gdal.GetDriverByName('GTiff').Create('local/rast.tif', cols, rows, 1, gdal.GDT_Float32)
    raster_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    raster_ds.SetProjection(EPGS)
    band = raster_ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    return np.array(band.ReadAsArray())

#####  Parallelizer
def parallelizerWithProcess(function, args:list, executors:int = 4):
    '''
    Parallelize the <function> in the input to the specified number of <executors>.
    @function: python function
    @args: list: list of argument to pas to the function in parallel. 
    '''
    with concurrent.futures.ProcessPoolExecutor(executors) as executor:
        # start_time = time.perf_counter()
        result = list(executor.map(function,args))
        # finish_time = time.perf_counter()
    # print(f"Program finished in {finish_time-start_time} seconds")
    print(result)

def parallelizerWithThread(function, args:list, executors:int = None):
    '''
    Parallelize the <function> in the input to the specified number of <executors>.
    @function: python function
    @args: list: list of argument to pas to the function in parallel. 
    '''
    with concurrent.futures.ThreadPoolExecutor(executors) as executor:
            start_time = time.perf_counter()
            result = list(executor.map(function, args))
            finish_time = time.perf_counter()
    print(f"Program finished in {finish_time-start_time} seconds")
    print(result)

def maxParalelizer(function,args):
    '''
    Same as paralelizer, but optimize the pool to the capacity of the current processor.
    NOTE: To be tested
    '''
    # print(args)
    pool = multiprocessing.Pool()
    result = pool.map(function,args)
    print(result)

### Interactives Visualization tools.
def genrateInteractivePlot(maskTifPath:os.path,probTifPath:os.path,outputPath:os.path, threshold:float=0.5)->None:
    '''
    Generates an interactive representation of flood susceptibility model's outputs. 

    @maskTifPath:os.path: Addres to the *.tif file containing the binary mask to be represented as references. 
    @probTifPath:os.path: Addres to the *.tif file containing the probabilities to be thresholdes dynamically on the interactive figure. 
    @outputPath:os.path: Path to save the interactive figure. 
    '''
    
    ## Import rasters as array
    mask_array = readRasterAsArry(maskTifPath)
    mask_array = (np.where(mask_array<0,0,mask_array)).astype(int)
    print(f'mask shape {mask_array.shape}')
    print(np.unique(mask_array))
    
    predictedProba = readRasterAsArry(probTifPath)
    shape = predictedProba.shape
    print(f'predictedProba shape {shape}')
    print(np.unique(predictedProba))
    
    mask_array = mask_array[0:shape[0],0:shape[1]]
    # Create initial images 
    output_file(outputPath)
    fig = figure()
    mask = ColumnDataSource(data={'mask': [mask_array]})
    # Crear la fuente de datos para la image

    tresholdedProba = (predictedProba>=threshold).astype(int)
    source = ColumnDataSource(data={'image': [tresholdedProba], 'initial_image': [predictedProba]})

    ## Plot ground truth mask. 
    fig.image(image='mask', source=mask, x=0, y=0, dw=shape[0], dh=shape[1], palette=["white",'#000000'])#,legend_label="Mask"

    ## Uncomment if desired  # Plot original prediction as a permanent reference.
    # fig.image(image='initial_image', source=source, x=0, y=0, dw=shape[0], dh=shape[1], palette=["white", '#000000'])

    ## Plot dinamic prediction. 
    fig.image(image='image', source=source, x=0, y=0, dw=shape[0], dh=shape[1], palette=["white", "red"], alpha=0.5,legend_label="Interactive prediction") # ,

    # Definir la barra deslizante (slider)
    slider = Slider(start=0, end=1, value=threshold, step=0.01, title="Threshold")

    # Create confusion matrix
    maskFlat = mask_array.flatten()#mask.data['mask']#
    probaFlat = tresholdedProba.flatten()#source.data['image']#
    mat = confusion_matrix(maskFlat,probaFlat)
    print(f"Original Conf Matrix : {mat}")
    
    confMat = createConfMatrixFig(mat) # plotInteractiveConfusionMatrix(mat)

    # Crear el layout
    layout = row(children=[column(fig,slider), confMat])# confMatrix,

    # Función para actualizar la imagen según el valor del slider
    update_image_js = CustomJS(args=dict(source=source, slider=slider, mask=mask), code="""
        const data = source.data['image'];
        const initial_data = source.data['initial_image'];
        const value = slider.value;
        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < data[i].length; j++) {
                if (initial_data[i][j] >= value) {
                    data[i][j] = 1;
                } else {
                    data[i][j] = 0;
                }
            }
        }
        source.change.emit();                         
    """
    )
    ## Capture the Slider.value changes to update the representation. 
    slider.js_on_change('value', update_image_js)

    ## Setting leyend
    # Personalizar la ubicación de la leyenda
    fig.legend.location = "top_right"
    # Cambiar la apariencia del texto de la leyenda
    fig.legend.label_text_font = "times"
    fig.legend.label_text_font_style = "italic"
    fig.legend.label_text_color = "navy"
    fig.legend.background_fill_alpha = 0
    
     # Guardar la figura HTML
    save(layout)
    
    # show(layout,sizing_mode="scale_width")
     
def createConfMatrixFig(matrix):
    # Create the Bokeh figure
    p = figure(title='Confusion Matrix', x_axis_label='Predicted Labels', y_axis_label='True Labels',
                x_range=(-0.5, 1.5), y_range=(-0.5, 1.5))
    p.width = 400
    p.height = 400

    # Create the color mapper with a predefined palette
    palette = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#eff3ff']
    mapper = linear_cmap(field_name='value', palette=palette, low=1, high=0)

    # Flatten the matrix and create ColumnDataSource
    matrix_Values = [matrix[0,0],matrix[0,1],matrix[1,0],matrix[1,1]]
    print(f"flat_matrix_Percent : {matrix_Values}")

    # Flatten the matrix (percentage) and create ColumnDataSource
    percentMAtrix =  matrix/np.sum(matrix)
    print(f"Percentage Conf Matrix : {percentMAtrix}")
    flat_matrix_Percent = [percentMAtrix[0,0],percentMAtrix[0,1],percentMAtrix[1,0],percentMAtrix[1,1]]
    print(f"flat_matrix_Percent : {flat_matrix_Percent}")
    
    actual= [0, 0, 1, 1]
    predicted= [0, 1, 0, 1]
    labels = ['True Neg', 'False Pos', 'False Neg','True Pos',]
    counts = [f"{value:.3f}" for value in flat_matrix_Percent]
    percentages = [f"{l}\n {(v)}\n {(p):.2%}" for l,v,p in zip(labels,matrix_Values,flat_matrix_Percent)]
    source = ColumnDataSource(data=dict(predicted=predicted, actual=actual, labels=labels, counts=counts,percentages=percentages, value=flat_matrix_Percent))

    # Add rectangles to the plot
    r = p.rect(x='predicted', y='actual', width=1, height=1, source=source,
            fill_color=mapper, line_width=1, line_color='black')

    # Add text annotations
    p.text(x='predicted', y='actual', text= 'percentages', source=source,
            text_align='center', text_baseline='middle', text_color='black')

    # Add color bar
    # color_bar = ColorBar(color_mapper=mapper['transform'], width=6, location=(0, 0),
                            # ticker=BasicTicker(desired_num_ticks=4))
    color_bar = r.construct_color_bar(width=10)  ## Equivalent to 2 previous lines
    
    p.add_layout(color_bar, 'right')


    # Customize plot
    p.xaxis.ticker = [0, 1]
    p.xaxis.major_label_overrides = {0: '0', 1: '1'}
    p.yaxis.ticker = [0, 1]
    p.yaxis.major_label_overrides = {0: '0', 1: '1'}
    
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 1
    # p.xaxis.major_label_orientation = np.pi / 3

    return p 


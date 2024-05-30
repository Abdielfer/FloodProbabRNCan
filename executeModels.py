
import myServices as ms
import models as m
import pandas as pd
import os, sys
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig,OmegaConf

def performeTraining(cfg:DictConfig,logManager:ms.logg_Manager):
    ''' 
    model, modelName, loss_fn,optimizer, labels, pathTrainingDSet,trainingParams, pathValidationDSet = None,scheduler = None, initWeightfunc= None, initWeightParams= None, removeCoordinates = True,logger:ms.logg_Manager = None)
    '''
    ## Current wdr
    orig_cwd = cfg.getHydra_Runtime_OutDir
    
    # Model
    modelChoice = OmegaConf.create(cfg.parameters['model'])
    model = instantiate(modelChoice)
    modelName = ms.makeNameByTime()
    logManager.update_logs({'model name': modelName})    
    logManager.update_logs({'model': modelChoice})    
    
    # Loss
    loss = OmegaConf.create(cfg.parameters['criterion'])
    loss_fn = instantiate(loss)
    logManager.update_logs({'loss function': loss})   
    
    # Optimizer & Scheduler
    optimizerOptions = OmegaConf.create(cfg.parameters['optimizer'])
    optimizer = instantiate(optimizerOptions)
    logManager.update_logs({'optimizer': optimizerOptions})   
    
    schedulOptions = OmegaConf.create(cfg.parameters['scheduler'])
    scheduler = instantiate(schedulOptions)
    logManager.update_logs({'schedule': schedulOptions})   
    
    # Labels
    Labels = cfg['targetColName']

    # Datasets definition  ... To be converted to a fucntion!!!!
    if cfg['combineDatsets']:
        cfg = combineDataset(cfg,logManager)
    
    pathTrainingDataset = cfg['trainingDataset']
    pathTestDataset = cfg['testDataset']
    
    # ### Logging dataset description as excell in the current working dir. 
    # descriptorExcellOutPath = os.path.join(orig_cwd,'trainingDatasetDescriptor.xlsx')
    # ms.reportDatasetDescrition(pathTrainingDataset,logManager,descriptorExcellOutPath)
    # ## ------
    # descriptorExcellOutPath = os.path.join(orig_cwd,'validationDatasetDatasetDescriptor.xlsx')
    # ms.reportDatasetDescrition( pathValidationDataset,logManager,descriptorExcellOutPath)
    # ## ------

    # Parameter initialization
    initWeightFunc = instantiate(OmegaConf.create(cfg.parameters['init_weight']))
    initWeightParams = cfg.parameters['initWeightParams']
    trainingParams = DictConfig(cfg.parameters)
    trainingParams.modelsFolder = cfg.modelsFolder
    if cfg['trackEvalPerfomance']:
        trainingParams.evalDatasetPath = cfg['evaluateInDataset']
        trainingParams.evalInBathches = cfg.parameters['evalInBathches']
    # print(f"Training params updated for modelsFolder {trainingParams}")
  
    # Modeling tool instanciate
    mlpc = m.ModelTrainCycle(model,modelName,loss_fn,optimizer,Labels,pathTrainingDataset,trainingParams,pathTestDataset,scheduler=scheduler,initWeightfunc=initWeightFunc, initWeightParams= initWeightParams,logger=logManager, nWorkers = cfg.nWorkers)
    
    # ## Excecute Training 
    model,metrics = mlpc.modelTrainer()
    
    ### Plot Confusion matrix from test dataset
    ConfMatfigName = str(modelName +'_TestConfMtrx.png')
    saveConfMtrx = os.path.join(orig_cwd,ConfMatfigName)
    mlpc.pltTestConfMatrx(saveTo = saveConfMtrx)
    
    ### Plot Losses
    lossesfigName = str(modelName +'_LossesVsEpocs.png')
    saveLosses = os.path.join(orig_cwd,lossesfigName)
    mlpc.plotLosses(showIt= False, saveTo = saveLosses)
    return model,metrics,modelName 

def performEvaluation(cfg:DictConfig,logManager:ms.logg_Manager,searchBestThreshold:bool=True):
    
    logManager.update_logs({' ': "------Evaluation process started------"})
   
    ## Model
    modelName = cfg.modelsFolder + cfg.evaluationModel + '.pkl'
    model =  ms.loadModel(modelName) 
    logManager.update_logs({'Model for evaluation': modelName})       
    
    ## Datasets
    evaluationDataset = cfg['evaluateInDataset']
    logManager.update_logs({'Evaluation Dataset': evaluationDataset}) 
    colsToDrop = cfg.parameters['colsToDrop']
    evalInBathches = cfg.parameters['evalInBathches']

    ## Evaluation
    labels = cfg['targetColName']
    threshold = cfg['sigmoidThreshold']

    X_DF, Y_real_DF = ms.importDataSet(evaluationDataset, labels, colsToDrop)
 
    Y_real,probs,y_hat = m.evaluateModel(X_DF,
                                        Y_real_DF, 
                                        model, 
                                        threshold, 
                                        evalInBathches)
    metrics = m.computeClassificationMetrics(Y_real,y_hat.cpu())
    logManager.update_logs({'Evaluation Metrics': metrics})
    
    if searchBestThreshold:
        # bestThresh, bestScore = m.exploreBestSigmoidThreshold_F1Score(Y_real,probs)
        # logManager.update_logs({'Best Sigmoid Threshold F1Score': bestThresh})
        # logManager.update_logs({'Best F1Macro': bestMacro})
        bestThreshMacro, bestMacro,bestScore = m.exploreBestSigmoidThreshold_F1Macro(Y_real,probs.cpu())
        logManager.update_logs({'Best Sigmoid Threshold': bestThreshMacro})
        logManager.update_logs({'Best F1_Macro_Average': bestMacro})
        logManager.update_logs({'Best F1Score for the best F1_Macro': bestScore})
        # Y_real,y_hat,probs,metrics = m.evaluateModel(model,evaluationDataset,labels=cfg['targetColName'],colsToDrop=colsToDrop,threshold=bestThresh)

    if cfg['createGISrepresentation']:
        prefixToPrint = cfg['evalPrefix']+'_'+ cfg.evaluationModel 
        print("------ Creating GIS representation --------")
        createGISRepresentation(cfg,evaluationDataset,y_hat.cpu(),prefix= prefixToPrint )
        
        prefixToPrintProbas = cfg['evalPrefix']+'_Probas_'+ cfg.evaluationModel 
        print("------ Creating GIS representation of probabilities --------")
        createGISRepresentation(cfg,evaluationDataset,probs.cpu(),prefix= prefixToPrintProbas )

    if cfg['createHTMLrepresentation']:
        print("------ Creating interactive.html map of prediction --------")
        prefixToPrint = cfg['evalPrefix']+ '_'+ cfg.evaluationModel 
        htmlOutFile = ms.replaceExtention(evaluationDataset,'.html')
        htmlOutFile = ms.addSubstringToName(htmlOutFile,prefixToPrint)
        createHTML_representation(cfg,evaluationDataset,probs.cpu(),outputHtmlImagePath=htmlOutFile)
    
    # Labels = cfg['targetColName']
    # computeFeatureImportance(evaluationDataset,Labels,colsToDrop,model)

    return Y_real,y_hat,metrics
   
def performeInference(cfg:DictConfig,logManager:ms.logg_Manager):
    ''' 
    Performe inference in the <inferInDataset> from <cfg> dictionary, applying the model in <inferenceModel>. 
    The output is a dataFrame with four columns <'x-coord,y_coord, y_val,Y_hat>, where y_val is the ground truth and y_hat is the infered value. 
    '''
    ## Current wdr
    orig_cwd = cfg.getHydra_Runtime_OutDir
    print(f"Origin wDir -> {orig_cwd}")
    logManager.update_logs({'Inference': "------Inference process started------"})
    ## Model
    modelName = cfg.modelsFolder + cfg.inferenceModel + '.pkl'
    model =  ms.loadModel(modelName) 
    logManager.update_logs({'model For inference': modelName})       
    
    ## Datasets
    inferenceDataset = cfg['inferInDataset']
    logManager.update_logs({'inference Dataset': inferenceDataset}) 
    
    ### Logging dataset description test
    # descriptorExcellOutPath = os.path.join(orig_cwd,'inferenceDatasetDescriptor.xlsx')
    ms.reportDatasetDescrition(inferenceDataset,logManager)

    # labels = cfg['targetColName']
    colsToDrop = list(cfg.parameters['colsToDrop'])
    print(colsToDrop)
    # labels = cfg['targetColName']
    # if labels not in colsToDrop:
    #     colsToDrop.append(labels)

    ## Inference
    threshold = cfg['sigmoidThreshold']
    inferInBatches = cfg.parameters['evalInBathches']
    y_hat,probs = m.inference(model,inferenceDataset,colsToDrop,threshold=threshold,inferInBatches=inferInBatches)
    # logManager.update_logs({'inference Metrics': metrics})
    # print(f"Unique values into probs {np.unique(probs.cpu())}")
    
    if cfg['createGISrepresentation']:
        prefixToPrint = cfg['inferPrefix']+'_'+ cfg.inferenceModel 
        print("------ Creating GIS representation --------")
        createGISRepresentation(cfg,inferenceDataset,y_hat.cpu(),prefix= prefixToPrint )
        prefixToPrintProbas = cfg['inferPrefix']+ '_Probas_'+ cfg.inferenceModel
        print("------ Creating GIS representation of probabilities --------")
        createGISRepresentation(cfg,inferenceDataset,probs.cpu(),prefix= prefixToPrintProbas )

    outputHtml = ms.replaceExtention(inferenceDataset,'.html')
    createHTML_representation(cfg,inferenceDataset,probs,prefix=cfg.inferenceModel,outputHtmlImagePath=outputHtml)
    
    return y_hat

def createHTML_representation(cfg:DictConfig,dataset,probVector,prefix:str='', outputHtmlImagePath:os.path =''):
    '''
  
    '''
    localShp =r'C:\Users\abfernan\CrossCanFloodMapping\GitLabRepos\FloodProbabRNCanAbd\local\temp.shp'
    localTiffOfProbas =r'C:\Users\abfernan\CrossCanFloodMapping\GitLabRepos\FloodProbabRNCanAbd\local\probas.tif'
    html_Image = ms.addSubstringToName(outputHtmlImagePath,prefix)
    # Create shpFile from inference.
    labels = cfg['targetColName'] 
    datasetToShpFile = ms.setInformationToCreateShpFile(dataset,probVector,labels)
    ms.buildShapefilePointFromCsvDataframe(datasetToShpFile,outShepfile=localShp)
    attribute = cfg['attribute']
    outputResolution = cfg['outputResolution']
    ms.rasterizePointsVector(localShp,localTiffOfProbas,attribute=attribute,pixel_size=outputResolution)
    os.remove(datasetToShpFile)
    os.remove(localShp)
    mask = cfg['mask4InteractRepresentation']
    ms.genrateInteractivePlot(mask,localTiffOfProbas,html_Image,threshold=cfg.sigmoidThreshold)
    # os.remove(localTiffOfProbas)

def createGISRepresentation(cfg:DictConfig, dataset, y_hat,prefix:str=""):
    ## Create shpFile from inference.
    labels = cfg['targetColName'] 
    datasetToShpFile = ms.setInformationToCreateShpFile(dataset,y_hat,labels)
    shpFile = ms.buildShapefilePointFromCsvDataframe(datasetToShpFile)
    rasterOut = ms.addSubstringToName(dataset, prefix) # TODO Remove Inference model as default!!!!!
    rasterOut = ms.replaceExtention(rasterOut,'.tif')
    attribute = cfg['attribute']
    outputResolution = cfg['outputResolution']
    ms.rasterizePointsVector(shpFile,rasterOut,attribute=attribute,pixel_size=outputResolution)
    return shpFile, rasterOut

def computeFeatureImportance(datasetPath, labels, colsToDrop, model):
    print('---------Starting feature importance ----------')
    X, Y = ms.importDataSet(datasetPath, labels, colsToDrop)
    IOU = m.intersection_over_union
    m.permutation_importance(X.values,Y.values,model,IOU)

def computeDatasetDistance(cfg:DictConfig,logManager:ms.logg_Manager):
    logManager.update_logs({'Process' : "--Measuring datasets similarity-----------"})
    colsToCompare = ['RelElev','HAND','proximity']
    logManager.update_logs({'colsToCompare' : colsToCompare})
    pathTrainingDataset = cfg['trainingDataset']
    logManager.update_logs({'training dataset' : pathTrainingDataset})
    pathValidationDataset = cfg['validationDataset']
    logManager.update_logs({'validationDataset' : pathValidationDataset})
    datasetA = pd.read_csv(pathTrainingDataset,index_col=None)
    datasetB = pd.read_csv(pathValidationDataset,index_col=None)
    _,mean=ms.datasetsEuclideanDistances(datasetA,datasetB,colsToCompare)
    logManager.update_logs({'Mean Euclidean Distance': mean})
    return mean

def buildDatasetsDistanceMatrix(datatsetCSVPathList:os.path,logManager:ms.logg_Manager):
    datasetslist = ms.createListFromCSV(datatsetCSVPathList)
    colsToCompare = ['RelElev','HAND','proximity']
    max_i = len(datasetslist)
    for i in range(max_i):
        logManager.update_logs({'New Cycle ' : "-----------------------------"})
        _,name,_ = ms.get_parenPath_name_ext(i)
        logManager.update_logs({'Dataset Reference ' : f"--------{name}--------"})

        for j in range(i+1,max_i):
            _,objectiveName,_ = ms.get_parenPath_name_ext(datasetslist[j])
            logManager.update_logs({'objectiveDataset' :f" ---> {objectiveName}"})
            datasetA = ms.normalizeDatasetByColNameList(datasetslist[i],colNamesList=colsToCompare)
            datasetB = ms.normalizeDatasetByColNameList(datasetslist[i],datasetslist[j],colNamesList=colsToCompare)           
            _,mean=ms.datasetsEuclideanDistances(datasetA,datasetB,colsToCompare)
            logManager.update_logs({'Mean Euclidean Distance': f" ==> {mean}"})
        break

## Combining Datasets:
def createCombinedDatasets_4RegModelingAppli(cfg:DictConfig, logManager:ms.logg_Manager):
    '''
    Create a balanced and stratified dataset from a list of dataset, to build a unified training set and test set. With this procedure, we ensure class balance and representativity between training-test. 
    '''  
    ### Log changes
    logManager.update_logs({'Combining datasets for training':'--------------------------------------------'})
    
    #####  Concat Datasets process 
    csvList = ms.createListFromCSV(cfg.TrainingDatasetsList)

    if cfg.combinedTrainSetPath:
        outTraining = cfg.combinedTrainSetPath
    else: 
        outTraining = ms.addSubstringToName(cfg.TrainingDatasetsList,'_CombinedTraining')

    if cfg.combinedTrainSetPath:
        outTest = cfg.combinedTestSetPath
    else: 
        outTest = ms.addSubstringToName(cfg.TrainingDatasetsList,'_CombinedTest')

    finalTrainingDataset = pd.DataFrame()
    finalTestDataset = pd.DataFrame()
    proportion = cfg.parameters['testSize']

    for i in csvList:
        ## Log the dataset's name to be combined
        _,name,_ = ms.get_parenPath_name_ext(i)
        logManager.update_logs({'->':f"{name}"})
        
        ### Compute balance-stratified undersampling: Ensure Balanced Clases and all datasets are represented in Training and validation
        X = pd.read_csv(i,index_col=None)
        Y = X['Labels']
        X.drop(['Labels'], axis=1, inplace = True)
        x_underSamp, y_undersamp = ms.randomUndersampling(X,Y)
        X_train, y_train, X_test, y_test = ms.stratifiedSplit(x_underSamp, y_undersamp, test_size=proportion)

        ## Concat training  ---
        X_train['Labels']= y_train
        finalTrainingDataset = pd.concat([finalTrainingDataset,X_train], ignore_index=True)
    
        ## Concat validation ----
        X_test['Labels']= y_test
        finalTestDataset = pd.concat([finalTestDataset,X_test], ignore_index=True)
    
    ## Log and save Training set
    description = finalTrainingDataset.describe()
    logManager.update_logs({'combined training dataset description \n': description})
    finalTrainingDataset.to_csv(outTraining,index=None)
   
    # Log and save validation set
    description = finalTestDataset.describe()
    logManager.update_logs({'combined test dataset description \n': description})
    finalTestDataset.to_csv(outTest,index=None)
    
    return outTraining,outTest

def normalizeByCombiningDataset_4RegModelingAppli( training:os.path, test:os.path,colsToNormalize:list = [])->os.path:
    trainingSet = pd.read_csv(training,index_col = None)
    testSet = pd.read_csv(test,index_col = None)
    # Create combine to compute globals min/max values.
    combineTrainValid = pd.concat([testSet,trainingSet], ignore_index=True)

    normalizedTraining = ms.normalizeDatasetByColNameList(combineTrainValid,trainingSet,colsToNormalize)
    normalizedTrainingPath = ms.addSubstringToName(training,'_Normal')
    normalizedTraining.to_csv(normalizedTrainingPath,index=None)

    normalizedTest = ms.normalizeDatasetByColNameList(combineTrainValid,testSet,colsToNormalize)
    normalizedTestPath = ms.addSubstringToName(test,'_Normal')
    normalizedTest.to_csv(normalizedTestPath,index=None)
    return normalizedTrainingPath,normalizedTestPath

def combineDataset(cfg:DictConfig, logManager:ms.logg_Manager):
    ## Combine the dataset in the cfg.TrainingDatasetsList to create stratified-balanced training/validation pairs. 
    pathTrainingDataset, pathTestDataset = createCombinedDatasets_4RegModelingAppli(cfg,logManager)
    if cfg.normalize:
        colsToNormalize = cfg['colsToNormalize']
        ## Normalize Datasets
        pathTrainingDataset_normal, pathTestDataset_normal = normalizeByCombiningDataset_4RegModelingAppli(pathTrainingDataset,pathTestDataset,colsToNormalize)
        cfg['trainingDataset'] = pathTrainingDataset_normal
        cfg['testDataset'] = pathTestDataset_normal
        if cfg.normalizeEvalSet:
            normalizeValidationSet(cfg, logManager, dataframeList=[pathTrainingDataset_normal,pathTestDataset_normal])  
    return cfg

def normalizeValidationSet(cfg:DictConfig, logManager:ms.logg_Manager, dataframeList = None):
    '''
    This fucntion normalize the validation dataset with resect a list of dataset, from where the min-max values per <colsToNormalize> is obtained. If no list is provided, the function consider the training and validation dataset concatenated, to extract the min-max per <colsToNormalize>. The function averwrite the value of <cfg['evaluateInDataset']> nd @return the updated <cfg> dictionary . 
    '''
    colsToNormalize = cfg['colsToNormalize']
    trainingParentDir,trainName,_=ms.get_parenPath_name_ext(cfg['trainingDataset'])
    if dataframeList is not None:
        dataframeList = dataframeList
    else:
        pathTrainingDataset = cfg['trainingDataset'] 
        pathTestDataset = cfg['testDataset']
        dataframeList = [pathTrainingDataset, pathTestDataset]
    fullDatset = ms.concatDatasets(dataframeList)
    print(fullDatset.describe())
    evaluationSetNormal = ms.normalizeDatasetByColNameList(fullDatset,cfg['evaluateInDataset'],colsToNormalize)
    evalDescritor = evaluationSetNormal.describe()
    logManager.update_logs({'Normalized validation dataset description \n': evalDescritor})

    _,evalName,ext =  ms.get_parenPath_name_ext(cfg['evaluateInDataset'])
    evaluationSetNormalPath = os.path.join(trainingParentDir,(evalName+"_normalWRT_"+trainName+ext))
    evaluationSetNormal.to_csv(evaluationSetNormalPath,index=None)
    cfg['evaluateInDataset'] = evaluationSetNormalPath 
    return cfg 

###  Main process. 
def updateConfigForHPC(cfg: DictConfig):
    args = cfg.hpcParameters
    if cfg['hpc']:
        print("----------HPC Training---------------")
        cfg = ms.updateHydraDicConfig(cfg,args)
    return cfg

def updateConfigForVelum(cfg: DictConfig):
    args = cfg.velumParameters
    if cfg['velum']:
        print("----------Velum Training---------------")
        cfg = ms.updateHydraDicConfig(cfg,args)
    return cfg

def mainProcess(cfg:DictConfig,logManager:ms.logg_Manager):
    ### Training 
    if cfg['performTraining']:
        print("-------------------   NEW Training -----------------------")
        _,_, modelName = performeTraining(cfg,logManager)
    ### Evaluation 
    if cfg['performEvaluation']:
        if cfg['evaluationModel']:
            ## Inference with the model defined in <cfg['inferenceModel']>
            logManager.update_logs({"____Evaluation in progress with Model": cfg['evaluationModel']})
            performEvaluation(cfg,logManager,searchBestThreshold=cfg['searchBestThreshold'])
        elif modelName is not None:
            ## Update the cfg dictionary with the last created model
            cfg['evaluationModel']= modelName
            logManager.update_logs({"____Evaluation in progress with Model": cfg['evaluationModel']})
            ## performe inference
            performEvaluation(cfg,logManager,searchBestThreshold=cfg['searchBestThreshold'])
        else:
            logManager.update_logs({'Evaluation called but not proced': "------No evaluation model defined ------"})
    else:
        logManager.update_logs({'Evaluation': "------No evaluation programed ------"})
    ### Inference
    if cfg['performInference']:
        if cfg['inferenceModel']:
            ## Inference with the model defined in <cfg['inferenceModel']>
            logManager.update_logs({"____Inference in progress with Model": cfg['inferenceModel']})
            performeInference(cfg,logManager)
            return
        elif modelName is not None:
            ## Update the cfg dictionary with the last created model
            cfg['inferenceModel']= modelName
            logManager.update_logs({"____Inference in progress with Model": cfg['inferenceModel']})
            performeInference(cfg,logManager)
        else:
            logManager.update_logs({'Inference called but not proced': "------No inference model defined ------"})       
    else:
        logManager.update_logs({'Inference': "------No inference programed ------"})

    ### Normalize Evaluation Dataset wrt train and validation concatenated. 
    if cfg['performTraining'] == False and cfg.normalizeEvalSet:
        logManager.update_logs({'Normalization': "------Normalizing evaluation dataset ------"})
        normalizeValidationSet(cfg, logManager)


@hydra.main(version_base=None,config_path=f"config", config_name="configMLPClassifier.yaml")
def main(cfg: DictConfig):
    ## Create logger
    logName = ms.getHydraJobName()
    logManager = ms.logg_Manager(logName = logName) 
    
    ## Set initial paths
    cfg['getHydra_Runtime_OutDir'] = ms.getHydra_Runtime_OutDir()
    logManager.update_logs({"Hydra_Runtime_OutDir":cfg['getHydra_Runtime_OutDir']})
    
    ## Update for Velum OR HPC
    if cfg['velum'] and cfg['hpc']:
        logManager.update_logs({"ERROR":"Both Velum and hpc updates are calls"})
        sys.exit("Both Velum and hpc path's updates are calls.")
    elif cfg['velum']:
        cfg = updateConfigForVelum(cfg)
    elif cfg['hpc']:
        cfg = updateConfigForHPC(cfg)

    mainProcess(cfg,logManager)

    # allBasinsTRainnigAOIList = '/home/abfernan/CrossCanFloodMap/A_DatasetsForMLP/RegionalModelingExplorationDatasets/IndividualsClass1ForRegModelingExpl/TrainingDatasets/TrainingDatasetProximityBufer2000m/trainingDatasetList_Bufered2000_Velum.csv'
    
    # buildDatasetsDistanceMatrix(allBasinsTRainnigAOIList,logManager)

if __name__ == "__main__":
    with ms.timeit():
        main()
       
        
    
   #######    Preprocessing For MLP
    # ### Concat all datasets with all classes
    # #####  Concat Datasets
    # csvList = ms.createListFromCSV(cfg.datasetsList)
    # out = ms.addSubstringToName(cfg.datasetsList,'_Concat')
    # ms.concatDatasets(csvList,out)

    # ### Clean Datasets from elevation < 0
    # outClean = ms.addSubstringToName(out,'_Clean')
    # DF = pd.read_csv(out,index_col=None)
    # DF = DF[DF.Cilp>=0]
    # print(DF.describe())
    # DF.to_csv(outClean, index=None)456+

    # ### Divide dataset by classes
    # csvClass1, csvClass5 = ms.extractFloodClassForMLP(outClean)
    
    # ### Compute balance undersampling for Class 1 
    # DF = pd.read_csv(csvClass1,index_col=None)
    # Y = DF['Labels']
    # DF.drop(['Labels'], axis=1, inplace = True)
    # x_underSamp, y_undersamp = ms.randomUndersampling(DF,Y)
    # print(len(x_underSamp), '---', len(y_undersamp))
    # x_underSamp['Labels']= y_undersamp
    # print(x_underSamp.head)
    # out = ms.addSubstringToName(csvClass1,'_Balance')
    # x_underSamp.to_csv(out, index=None)
    
    # ### Compute balance undersampling for Class 5 
    # DF = pd.read_csv(csvClass5,index_col=None)
    # Y = DF['Labels']
    # DF.drop(['Labels'], axis=1, inplace = True)
    # x_underSamp, y_undersamp = ms.randomUndersampling(DF,Y)
    # print(len(x_underSamp), '---', len(y_undersamp))
    # x_underSamp['Labels']= y_undersamp
    # print(x_underSamp.head)
    # out = ms.addSubstringToName(csvClass5,'_Balance')
    # x_underSamp.to_csv(out, index=None)
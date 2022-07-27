import pandas as pd
import myServices as ms
import randForest as r
import hydra
from omegaconf import DictConfig

def executeRFRegressor(cfg: DictConfig):
    log = {}
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    # penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    rForestReg = r.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, bestParameters, r2_validation= rForestReg.fitRFRegressorGSearch()
    r.saveModel(best_estimator,name)
    _,x_validation,_,_ = rForestReg.getSplitedDataset()
    regressorName, featureImportance = r.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2_validation)
    # Fill Log
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = regressorName
    log['best_param'] = bestParameters
    log['features_Importance'] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['r2_score'] = r2_validation 
    return best_estimator,name,log


def executeRFRegressorWeighted(cfg: DictConfig):
    log = {}
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    penalty = cfg.weightPenalty  ## UNCOMENT Onli for weighted fit
    arg = cfg.parameters
    rForestReg = r.implementRandomForestRegressor(pathDataset,'percentage', percentOfValidation, arg)
    best_estimator, bestParameters, r2_validation= rForestReg.fitRFRegressorWeighted(penalty)
    r.saveModel(best_estimator,name)
    _,x_validation,_,_ = rForestReg.getSplitedDataset()
    regressorName, featureImportance = r.investigateFeatureImportance(best_estimator,name,x_validation)
    print(r2_validation)
    # Fill Log
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = regressorName
    log['best_param'] = bestParameters
    log['features_Importance'] = pd.DataFrame(featureImportance).to_string(justify= 'left')
    log['r2_score'] = r2_validation 
    return best_estimator,name,log

def executeRFCalssifier(cfg: DictConfig):
    log = {}
    name = r.makeNameByTime()
    local = cfg.local
    pathDataset = local + cfg['pathDataset']
    percentOfValidation = cfg.percentOfValidation
    arg = cfg.parameters
    rfClassifier = r.implementRandomForestCalssifier(pathDataset,'percentage', percentOfValidation, arg)
    _,x_validation,_,_ = rfClassifier.getSplitedDataset()
    best_estimator, bestParameters = rfClassifier.fitRFClassifierGSearch()
    classifierName, featureImportance = r.investigateFeatureImportance(best_estimator,name,x_validation)
    accScore, macro_averaged_f1, micro_averaged_f1, ROC_AUC_multiClass = rfClassifier.computeClassificationMetrics(best_estimator)
    log['dataset'] = cfg['pathDataset']
    log['model_Id'] = name
    log['regressor_Name'] = classifierName
    log['best_param'] = bestParameters
    log['features_Importance'] = featureImportance.to_dict('tight')
    log['Accuraci_score'] = accScore
    log['macro_averaged_f1'] = macro_averaged_f1
    log['micro_averaged_f1'] = micro_averaged_f1
    log['ROC_AUC_multiClass'] = ROC_AUC_multiClass
    return best_estimator,name,log



@hydra.main(config_path=f"config", config_name="config.yaml")
def main(cfg: DictConfig):
    best_estimator,name, log = executeRFRegressorWeighted(cfg)
    r.saveModel(best_estimator, name)
    logToSave = pd.DataFrame.from_dict(log, orient='index')
    logToSave.to_csv(name +'.csv',index = True, header=True) 

if __name__ == "__main__":
    with ms.timeit():
        main()
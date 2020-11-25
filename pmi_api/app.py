import logging
import json
import math
import time
import os
import re
import ntpath
import pandas as pd
import subprocess
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as sk_auc
import matplotlib.pyplot as plt
from azure.storage.blob import BlobClient
from azure.storage.blob import ContainerClient
from flask import Flask, jsonify, request

def upload_to_blob(conn_str,container_name,local_name,blob_name):
    blob = BlobClient.from_connection_string(conn_str=conn_str,container_name=container_name, blob_name=blob_name)
    with open(local_name, "rb") as data:
        blob.upload_blob(data,overwrite=True)

def download_blob(conn_str,container_name,local_name,blob_name):
    blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=container_name, blob_name=blob_name)
    with open(local_name, "wb") as my_blob:
        logging.info(blob_name)
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)

def upload_folder_to_blob(name,folder_path,container_name,conn_str):
    
    base = ntpath.basename(name)
    path_remove = re.sub(base,'',name)
    local_path = name

    for r,d,f in os.walk(local_path):        
        if f:
            for file in f:
                file_path_on_azure = os.path.join(r,file).replace(path_remove,"")
                file_path_on_local = os.path.join(r,file)
                if folder_path != '':
                    blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=container_name, blob_name=folder_path +'/'+file_path_on_azure)
                else :
                    blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=container_name, blob_name=file_path_on_azure)
                with open(file_path_on_local, "rb") as data:
                    blob.upload_blob(data,overwrite=True)
    return

def data_for_model(trainfile,bkg_file,cat_file):
    with open(bkg_file, 'w', encoding='utf-8') as bkg:
        with open(cat_file, 'w', encoding='utf-8') as cat:
            # data = pd.read_table(trainfile,sep='\t',names=['Label','Id','Text'])
            data = pd.read_csv(trainfile,sep='\t',names=['Label','Id','Text'])
            data = data.to_dict('records')
            for record in data:
                if str(record['Label']) == '1':
                    bkg.write(str(record['Label'])+'\t'+str(record['Id'])+'\t'+str(record['Text'])+'\n')
                    cat.write(str(record['Label'])+'\t'+str(record['Id'])+'\t'+str(record['Text'])+'\n')
                elif str(record['Label']) == '0':
                    bkg.write(str(record['Label'])+'\t'+str(record['Id'])+'\t'+str(record['Text'])+'\n')
                # elif str(record['Label']) == 'none' or str(record['Label']) == 'None' or str(record['Label']) =='':
                #     bkg.write(str(record['Label'])+'\t'+str(record['Id'])+'\t'+record['Text']+'\n')
    return

def accuracy_anasysis(df,fprate):
    df = df.sort_values(by = 'Score', ascending = False)
    #count for 
    TP=0
    FP=0
    FPR=0
    count_0 = len(df['Label'])-sum(df['Label'])
    count_1 = sum(df['Label'])
    #ccalculate FP rate
    i = 0
    for i in range(len(df)):
        if FP<fprate*count_0: #1% for offensive 0.1% for source code model
            label = df.iloc[i]['Label']
            threshold = df.iloc[i]['Score']
            FPR = FP/count_0
            if label == 1:
                TP+=1
            if label==0:
                FP+=1
            
        
    Recall = TP/count_1
    TN =count_0-FP
    FN =count_1 - TP

    print('Recall:', Recall,'FPR:', FPR, 'TP:',TP, 'threshold:', threshold,'FP:',FP,'TN:',TN,'FN:',FN)
    return threshold

def plot_roc_score(label_test, score,name):
    x, y, _ = roc_curve(label_test, score)
    roc_auc = sk_auc(x, y)
    
    plt.figure()
    lw = 2
    plt.plot(x, y, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # name = re.sub('.tsv','_roc.png',name)
    name = name+'/roc.png'
    plt.savefig(name)
    plt.show()
    return

app = Flask(__name__)
 
@app.route('/',methods = ['GET'])
def entrypoint():
    return"This is the sample api service"
 
@app.route('/pmi/', methods = ['POST'])

def pmi():
    print("pmi function")
    req_body = request.get_json()
    if req_body =='':
        return "Pass the query string is empty"
    # fil = "C:/Users/v-saigoteti/Desktop/pmi"
    pmi_path = "pmi/"
    library_path = "library/"
    if not os.path.exists(pmi_path):
        os.mkdir(pmi_path)
    library_files = os.listdir(library_path)
    connect_str = "DefaultEndpointsProtocol=https;AccountName=amlpracticeblob;AccountKey=KNSLdoR3zy2GMjNbpSXBym1xgMk6J6CoE642fRmcW3fhdVZKVoXZgO/T5gzkCzS6YeN23XBdbkqkmE+OWCJiCg==;EndpointSuffix=core.windows.net;"
    container_name = "pmi"
    container_client = ContainerClient.from_connection_string(connect_str,container_name)
    blob_list = container_client.list_blobs()
    blob_names=[]
    for blob in blob_list:
        blob_names.append(blob.name)
    # for i in blob_names:
    #     if "amd64" in i:
    #         blob_name = library_path+ntpath.basename(i)
    #         download_blob(connect_str,container_name,blob_name,i)
    # logging.info(os.listdir(library_path))

    if req_body["processType"] == "Training" :

        data_path = req_body['containerPath']
        data_path_local = pmi_path #+'data'
        
        train_data_local = data_path_local+'train'
        validation_data_local = data_path_local+'validation'
        test_data_local = data_path_local+'test'

        if not os.path.exists(data_path_local):
            os.mkdir(data_path_local)
        if not os.path.exists(train_data_local):
            os.mkdir(train_data_local)
        if not os.path.exists(validation_data_local):
            os.mkdir(validation_data_local)
        if not os.path.exists(test_data_local):
            os.mkdir(test_data_local)
        
        logging.info("downloading data")
        for i in blob_names:
            if data_path+'/train.txt' in i :
                blob_name = train_data_local+'/'+ntpath.basename(i)
                train_file = blob_name
                download_blob(connect_str,container_name,blob_name,i)
            elif data_path+'/validation.txt' in i :
                blob_name = validation_data_local+'/'+ntpath.basename(i)
                validation_file = blob_name
                download_blob(connect_str,container_name,blob_name,i)
            elif data_path+'/test.txt' in i :
                blob_name = test_data_local+'/'+'data_harassment_test.txt'
                test_file = blob_name
                download_blob(connect_str,container_name,blob_name,i)
        logging.info(os.listdir(train_data_local))
        logging.info(os.listdir(test_data_local))
        background_path = data_path_local+"/bkd_data"
        category_path = data_path_local+"/cat_data"
        if not os.path.exists(background_path):
            os.mkdir(background_path)
        if not os.path.exists(category_path):
            os.mkdir(category_path)
        logging.info(len(os.listdir(background_path)))
        data_for_model(train_file,background_path+'/data_discrimination_bkg.txt',category_path+'/data_discrimination_cat.txt')
        argument = ' -backgroundTraining -bkgPath '+'"'+background_path+'"'
        try :
            # for i in blob_names:
            #     if "amd64" in i:
            #         blob_name = library_path+ntpath.basename(i)
            #         download_blob(connect_str,container_name,blob_name,i)
            # logging.info(os.listdir(library_path))
            if os.path.exists(library_path+'Microsoft.Office.Compliance.MATagging.Tools.PMIExperimentTool.exe'):
                # os.chmod(library_path+'Microsoft.Office.Compliance.MATagging.Tools.PMIExperimentTool.exe', 0o777)
                process = library_path+'Microsoft.Office.Compliance.MATagging.Tools.PMIExperimentTool.exe'+argument
                logging.info(process)
                logging.info("starting the training process")
                # args = shlex.split(process)
                # inp = subprocess.check_output(args)
                inp = subprocess.call(process)
                # logging.info(inp)
                # subprocess.Popen(args)
                # subprocess.run(args)
                # a = os.system(process)
                # logging.info(a)
                # if a!=0:
                #     return "process exited"
        except Exception as e:
            logging.info(e)
            logging.info("Failed to create background file")
            return "Failed to create background file"
        bak_file_path = os.listdir(background_path+"_PMIExperiment")
        bak_file_path = background_path+"_PMIExperiment/"+bak_file_path[0]
        bak_file = os.listdir(bak_file_path)
        bak_file_path = bak_file_path+'/'+bak_file[0]
        argument = ' -categoryTraining -bkgPath "'+bak_file_path+'"'
        argument = argument+' -trainPath "'+category_path+'"'
        try :
            process = library_path+'Microsoft.Office.Compliance.MATagging.Tools.PMIExperimentTool.exe'+argument
            logging.info(process)
            logging.info("starting the training process")
            # args = shlex.split(process)
            subprocess.call(process)
            # subprocess.run(args)
            # os.system(process)
            local_name = os.listdir(category_path+"_PMIExperiment")[0]
            local_path = category_path+"_PMIExperiment/"+local_name
            model_upload = category_path+"_PMIExperiment/"
            local_name = os.listdir(category_path+"_PMIExperiment/"+local_name)[0]
            blob_name = "models/"+local_name
            local_name = local_path+'/'+local_name
            model_path = local_path
            # upload_to_blob(connect_str,container_name,local_name,blob_name)
        except Exception as e:
            logging.info("Failed to create model file")
            return "Failed to create model file"

        argument = ' -predict -modelsFilePath "'+model_path+'" -dataFolderPath "'+test_data_local+'"'
        try :
            process = library_path+'Microsoft.Office.Compliance.MATagging.Tools.PMIExperimentTool.exe'+argument
            process = process.replace('/','\\')
            logging.info(process)
            logging.info("starting the testing process")
            # os.system(process)
            # args = shlex.split(process)
            # subprocess.Popen(args)
            subprocess.call(process)
            local_path = test_data_local+'_PMIExperiment/'
            test_upload = local_path
            local_name = os.listdir(test_data_local+'_PMIExperiment/')[0]
            local_name = local_path+local_name+'/'+'summary.txt'
            test_result = local_name
            blob_name = 'results/summary.txt'
            # upload_to_blob(connect_str,container_name,local_name,blob_name)

            metrics_path = data_path_local+"metrics"
            if not os.path.exists(metrics_path):
                os.mkdir(metrics_path)
            df_test = pd.read_table(test_file,sep='\t',names = ['Label','ID','Text'])
            df_test_predictions = pd.read_table(test_result,sep='\t',names = ['Label','Score','zero','ID'])
            ids2 = list(df_test_predictions['ID'])
            ids1 = list(df_test['ID'])
            df_test.set_index('ID',inplace = True)
            exclude = []
            for i in range(len(ids1)):
                if ids1[i] not in ids2:
                    try:
                        df_test = df_test.drop(ids1[i])
                    except :
                        exclude.append(ids1[i])
            plot_roc_score(df_test['Label'],df_test_predictions['Label'],metrics_path)
            classification_report_test = classification_report(df_test['Label'], df_test_predictions['Label'])
            confusion_matrix_test = confusion_matrix(df_test['Label'], df_test_predictions['Label'])
            with open(metrics_path+'/'+'classification_report_test'+'.txt','w') as f:
                f.write(classification_report_test)
            plt.matshow(confusion_matrix_test)
            plt.title('Confusion Matrix Test')
            plt.colorbar()
            plt.savefig(metrics_path+'/'+'confusion_matrix_test'+'.png')
            logging.info("finished executing pmi")
            upload_folder_to_blob(metrics_path,data_path,container_name,connect_str)
            upload_folder_to_blob(test_upload,data_path,container_name,connect_str)
            upload_folder_to_blob(model_upload,data_path,container_name,connect_str)
            return "model trained"
        except Exception as e:
            logging.info("Failed to test file")
            return "Failed to test file"

    elif req_body["processType"] == "Polish" :
        models_path = pmi_path+"models/"
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        model_path = req_body['containerPath']
        for i in blob_names:
            if model_path in i:
                blob_name = models_path+ntpath.basename(i)
                download_blob(connect_str,container_name,blob_name,i)
        argument = " -polishModel "
        if "predictionThreshold" in req_body:
            predictionthreshold = req_body["predictionThreshold"]
            argument = argument+'-predictionThreshold '+predictionthreshold+' '
        if "docTokenCountThreshold" in req_body:
            docTokenCountThreshold = req_body["docTokenCountThreshold"]
            argument = argument+'-docTokenCountThreshold '+docTokenCountThreshold+' '
        if "topicId" in req_body:
            topicid = req_body["topicId"]
            argument = argument+'-topicId "'+topicid+'" '
        if "topicName" in req_body:
            topicname = req_body["topicName"]
            argument = argument+'-topicName "'+topicname+'" '
        argument = argument + "-modelFilePath "+ '"'+blob_name+'"'
        try :
            process = library_path+'Microsoft.Office.Compliance.MATagging.Tools.ExperimentUtilityTool.exe'+argument
            logging.info(process)
            logging.info("starting the reduction process")
            os.system(process)
            local_files = os.listdir(models_path+'Polished/')
            parent_path = model_path.replace(ntpath.basename(model_path),'')
            for i in local_files:
                blob_name = parent_path+'polished/'+i
                local_name = models_path+'Polished/'+i
                upload_to_blob(connect_str,container_name,local_name,blob_name)
            return "model polished"
        except Exception as e:
            logging.info("Failed to background file")
            return "Failed to background file"

    elif req_body["processType"] == "Reduce size" :
        models_path = pmi_path+"models/"
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        model_path = req_body['containerPath']
        for i in blob_names:
            if model_path in i:
                blob_name = models_path+ntpath.basename(i)
                download_blob(connect_str,container_name,blob_name,i)
        argument = " -reducePMIModelSize "
        if "termLimit" in req_body:
            termlimit = req_body["termLimit"]
            argument = argument+'-termLimit '+termlimit+' '
        if "minCount" in req_body:
            mincount = req_body["minCount"]
            argument = argument+'-minCount '+mincount+' '
        argument = argument + "-modelFilePath "+ '"'+blob_name+'"'
        try :
            process = library_path+'Microsoft.Office.Compliance.MATagging.Tools.ExperimentUtilityTool.exe'+argument
            logging.info(process)
            logging.info("starting the reduction process")
            os.system(process)
            local_files = os.listdir(models_path+'ReducedSize/')
            parent_path = model_path.replace(ntpath.basename(model_path),'')
            for i in local_files:
                blob_name = parent_path+'ReducedSize/'+i
                local_name = models_path+'ReducedSize/'+i
                upload_to_blob(connect_str,container_name,local_name,blob_name)
            return "model size reduced"
        except Exception as e:
            logging.info("Failed to background file")
            return "Failed to background file"
    return "PMI executed"

if __name__ == '__main__': 
    app.run(debug = True) 
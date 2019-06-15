#!/usr/bin/env python
# coding: utf-8

# ## Importando os pacotes a serem utilizados

# In[1]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark import HiveContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics as metric
#from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler,OneHotEncoder
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler,OneHotEncoder
from pyspark.ml import Pipeline

if __name__ == "__main__":
# ### Inicio integração Hive

# In[2]:


    spark.sparkContext._conf.getAll()


# In[3]:


    conf = spark.sparkContext._conf.setAll([
        ("hive.metastore.uris", "thrift://localhost:9083")])


# In[4]:


    spark.stop()


# In[5]:


    sc = SparkContext()


# In[6]:


    spark = SparkSession.builder.config(conf=conf).getOrCreate()


# In[7]:


    spark.sparkContext._conf.getAll()


# In[8]:


    df = spark.sql("SHOW TABLES")
    df.show()


# ### FIM integração Hive

# In[9]:


# Importando a base do HDFS
    file_location = 'hdfs:///user/labdata/marketing_data'
    file_type = "csv"

# schema CSV 
    infer_schema = "true"
    first_row_is_header = "false"
    delimiter = ","

# As opções aplicadas são para arquivos CSV. Para outros tipos de arquivo, estes serão ignorados.
    df_marketing_data = spark.read.format(file_type)   .option("inferSchema", infer_schema)   .option("header", first_row_is_header)   .option("sep", delimiter)   .load(file_location)


# In[10]:


# Inserindo cabeçalho (a tabela do Mysql está sem o cabeçalho)
    DefColumnNames=df_marketing_data.schema.names
    HeaderNames=['age','job','marital','education','default','housing','loan','contact','month','day_of_week','duration','campaign','pdays','previous','poutcome','emp_var_rate','cons_price_idx','cons_conf_idx','euribor3m','nr_employed','y']

    for Idx in range(0,21):
        df_marketing_data=df_marketing_data.withColumnRenamed(DefColumnNames[Idx],HeaderNames[Idx])
# Retirando a variável duration pois interfere na previsão do modelo
        df_marketing_data = df_marketing_data.drop ('duration')


# In[11]:


###    df_marketing_data.printSchema()


# In[12]:


# Definindo variáveis categóricas
    categoricalColumns = []
    numericCols = []
    for i in df_marketing_data.dtypes:
        if i[1]=='string':
            categoricalColumns  += [i[0]]
        elif i[1]=='int' or i[1]=='double':
            numericCols  += [i[0]]

###    print(categoricalColumns)
###    print(numericCols)


# In[13]:


# Tratamento das colunas categóricas usando StringIndex / Encoder
# StringIndexer -> Designa um Índice para cada Categoria em uma Variável Categórica
# OnHotEncoder -> Converte as Variáveis Categóricas em Vetores que aprimoram o processo de predição
    stages = [] 
    for categoricalCol in categoricalColumns:
      stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol+"Index")
      encoder = OneHotEncoder(inputCol=categoricalCol+"Index", outputCol=categoricalCol+"classVec")
      stages += [stringIndexer, encoder]
  
    label_stringIdx = StringIndexer(inputCol = "y", outputCol = "label")
    stages += [label_stringIdx]


# In[14]:


## Vector Assembler Criação de um vetor com Todas as Variáveis
    assemblerInputs = ['jobclassVec', 'maritalclassVec', 'educationclassVec', 'defaultclassVec', 'housingclassVec', 'loanclassVec', 'contactclassVec', 'monthclassVec','poutcomeclassVec'] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]


# In[15]:


## PipeLine , utilizado para aplicar todas as transformações no DataFrame
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(df_marketing_data)
    df_marketing_data_prep = pipelineModel.transform(df_marketing_data)
###    df_marketing_data_prep.printSchema()


# In[16]:


    df_marketing_data_prep.take(5)


# In[17]:


#Configurando o modelo para 100 iteracoes
    from pyspark.ml.classification import GBTClassifier, GBTClassificationModel
    modelo = GBTClassifier(labelCol="label", featuresCol="features", maxIter=100)


# In[18]:


# Divisão dos Dados de Teste e Treino
    (marketing_model_treino, marketing_model_teste) = df_marketing_data_prep.randomSplit([0.7, 0.3])


# In[19]:


# Preparando o Treino
    modelo_treino = modelo.fit(marketing_model_treino)
    modelo_treino.featureImportances


# In[28]:


# Salvando o modelo no HDFS
    hdfs_path = "/user/labdata/modelo_BST2"
    modelo_treino.write().overwrite().save(hdfs_path)


# In[29]:


# Carga do Modelo Treinado
    modelo_treino2 =GBTClassificationModel.load(hdfs_path)


# In[30]:


# Predição do modelo
    predict = modelo_treino2.transform(marketing_model_treino)


# In[31]:


    predict.select("features").take(5)


# In[32]:


    predict.show()


# In[33]:


    results = predict.select(['probability', 'label'])


# In[34]:


# Salvando modelo no Hive
    import pyspark
    df_writer = pyspark.sql.DataFrameWriter(predict)
    df_writer.saveAsTable('default.boosting_output', format='parquet', mode='overwrite')


# In[35]:


#
#spark.sql("SELECT * FROM default.boosting_output").show()


# In[36]:


    results_collect = results.collect()
    results_list = [(float(i[0][0]), 1.0-float(i[1])) for i in results_collect]
    scoreAndLabels = sc.parallelize(results_list)


# In[37]:


    metrics = metric(scoreAndLabels)
    print("The ROC score is (@maxIter=100): ", metrics.areaUnderROC)


# In[38]:


#    from sklearn.metrics import roc_curve, auc
#    from matplotlib import pyplot as plt
# 
#    fpr = dict()
#    tpr = dict()
#    roc_auc = dict()
# 
#    y_test = [i[1] for i in results_list]
#    y_score = [i[0] for i in results_list]
# 
#    fpr, tpr, _ = roc_curve(y_test, y_score)
#    roc_auc = auc(fpr, tpr)
# 
#    get_ipython().run_line_magic('matplotlib', 'inline')
#    plt.figure()
#  plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#    plt.plot([0, 1], [0, 1], 'k--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
#    plt.legend(loc="lower right")
#    plt.show()
#    display()


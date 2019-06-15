# FIA PROJETO FINAL - TURMA 2
# Script para a exportação da tabela DataBase Marketing Additional para o HDFS
# Database Origem: Mysql , Marketing 
# Tabela: BANK_MARKETING
# Destino: HDFS / marketing_data

sqoop import \
   --connect jdbc:mysql://elephant/marketing \
   --username root \
   --table BANK_MARKETING \
   --target-dir marketing_data -m 1 \

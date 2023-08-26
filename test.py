from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader, TextLoader


loader = CSVLoader(file_path='./names.csv', csv_args={'delimiter': ','}, source_column='Name')
data = loader.load()

#pdfloader = PyPDFLoader(file_path='.\\INTRODUCTORY_ENGINEERING.pdf')

#data = pdfloader.load()
print(data)
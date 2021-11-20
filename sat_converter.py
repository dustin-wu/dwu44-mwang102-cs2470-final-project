import PyPDF2
 
# creating a pdf file object 
pdfFileObj = open('act_data/pdfs/ACT_Test_2005-06.pdf', 'rb') 
    
# creating a pdf reader object 
pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 
    
# printing number of pages in pdf file 
print(pdfReader.numPages) 
    
content = ''
for page in range(pdfReader.numPages):
	# creating a page object
	pageObj = pdfReader.getPage(page)
	# extracting text from page 
	content += pageObj.extractText() + "\n"
    

with open('test1.txt', 'w') as f:
    f.write(content)

    
# closing the pdf file object 
pdfFileObj.close()
import collections
import math
import os
import re
import codecs

stop_words = ["a","about","above","after","again","against","all","am","an","and",
"any","are","aren't","as","at","be","because","been","before","being","below",
"between","both","but","by","can't","cannot","could","couldn't","did","didn't",
"do","does","doesn't","doing","don't","down","during","each","few","for","from",
"further","had","hadn't","has","hasn't","have","haven't","having","he","he'd",
"he'll","he's","her","here","here's","hers","herself","him","himself","his","how",
"how's","i","i'd","i'll","i'm","i've","if","in","into","is","isn't","it","it's","its",
"itself","let's","me","more","most","mustn't","my","myself","no","nor","not","of",
"off","on","once","only","or","other","ought","our","ours","ourselves","out","over",
"own","same","shan't","she","she'd","she'll","she's","should","shouldn't","so","some",
"such","than","that","that's","the","their","theirs","them","themselves","then","there",
"there's","these","they","they'd","they'll","they're","they've","this","those","through",
"to","too","under","until","up","very","was","wasn't","we","we'd","we'll","we're","we've",
"were","weren't","what","what's","when","when's","where","where's","which","while","who",
"who's","whom","why","why's","with","won't","would","wouldn't","you","you'd","you'll",
"you're","you've","your","yours","yourself","yourselves"]

def ReadFile(file,filepath):
    fileHandler = codecs.open(filepath+"\\" + file,'rU','latin-1')
    Findwords = re.findall('[A-Za-z0-9\']+', fileHandler.read())
    allwords = list()
    for word in Findwords:
        word = word.lower()
        allwords+=[word]
    fileHandler.close()    
    return allwords
    
def numberofFiles(filepath):
    wordList = list()
    NumberOfFiles = 0
    for files in os.listdir(filepath):
        if files.endswith(".txt"):
            wordList += ReadFile(files,filepath)
            NumberOfFiles+=1
    return wordList, NumberOfFiles

train_path = 'C:/Users/shash/Desktop/MS CE/Subject/Spring 2020/CS6375/CS-6375-Machine-Learning-master/Assignment 3/train'
hamPath = train_path + '/ham'
spamPath = train_path + '/spam'

hamCount, spamCount = 0, 0
hamWord = []
spamWord = []
hamWord, hamCount = numberofFiles(hamPath)
spamWord, spamCount = numberofFiles(spamPath)

def findPhamorPspam(HamOrSpam):
    """
    Input: String to choose between Ham and Spam
    Output: Real Number from 0 to 1 giving the probability
    """
    if HamOrSpam == "spam":
        spamProb = spamCount/(spamCount + hamCount)
        return spamProb
    else:
        hamProb = hamCount/(spamCount + hamCount)
        return hamProb

#Using Collections Counter to count the number of occurences of each words
hamDictionary = dict(collections.Counter(word.lower() for word in hamWord))
spamDictionary = dict(collections.Counter(word.lower() for word in spamWord))

#Extracting the Unique Words
bagofWords = hamWord + spamWord
bagofwordsDict = collections.Counter(bagofWords)

def missingWords(AllWords,HamSpamWords):
    for words in AllWords:
        if words not in HamSpamWords:
            HamSpamWords[words] = 0
            
missingWords(bagofwordsDict,hamDictionary)
missingWords(bagofwordsDict,spamDictionary)

#Define two dictionaries to compute the probability of spam and ham words
probofHam = dict()
probofSpam = dict()

def FindProbabilityOfWord(classifier,removestopwords):
    Counter = 0
    if(removestopwords ==1):
            for word in stop_words:
                if word in hamDictionary:
                    del hamDictionary[word]
                if word in spamDictionary:
                    del spamDictionary[word]
                if word in bagofwordsDict:
                    del bagofwordsDict[word]                    
    if classifier == "ham":
        """
        Takes the ham or spam and uses the laplacian smoothing
        """
        for word in hamDictionary:
            Counter += (hamDictionary[word] + 1)
        for word in hamDictionary:
            probofHam[word] = math.log((hamDictionary[word] + 1)/(Counter + len(bagofwordsDict)),2)
    elif classifier == "spam":
        for word in spamDictionary:
            Counter += (spamDictionary[word] + 1)
        for word in spamDictionary:
            probofSpam[word] = math.log((spamDictionary[word] + 1)/(Counter + len(bagofwordsDict)),2) 
            
#caluculating probability for each word in ham and Spam folders 
FindProbabilityOfWord("ham",0)
FindProbabilityOfWord("spam",0) 


#Finally classify the emails as ham or spam    
def predicthamorSpam(pathToFile, classifier):
    ProbabilityOfHam = 0 
    ProbabilityOfSpam = 0 
    InCorrectlyClassified = 0
    NumberOfFiles = 0
                   
    if classifier == "spam":
        for fileName in os.listdir(pathToFile):
            words =ReadFile(fileName,pathToFile)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ProbabilityOfHam = math.log(findPhamorPspam("ham"),2)
            ProbabilityOfSpam = math.log(findPhamorPspam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + .... 
            for word in words:
                if word in probofHam:
                    ProbabilityOfHam += probofHam[word]
                if word in probofSpam:
                    ProbabilityOfSpam += probofSpam[word]
            NumberOfFiles +=1
            if(ProbabilityOfHam >= ProbabilityOfSpam):
                InCorrectlyClassified+=1
    if classifier == "ham":
        for fileName in os.listdir(pathToFile):
            words =ReadFile(fileName,pathToFile)
            #find actual P(ham) and P(spam) i.e. (number of ham documents / Total no of documents)
            ProbabilityOfHam = math.log(findPhamorPspam("ham"),2)
            ProbabilityOfSpam = math.log(findPhamorPspam("spam"),2)
            #log(P(ham|bodyText)) = log(P(ham)) + log(P(word1|ham)) + log(P(word2|ham)) + ....            
            for word in words:
                if word in probofHam:
                    ProbabilityOfHam += probofHam[word]
                if word in probofSpam:
                    ProbabilityOfSpam += probofSpam[word]
            NumberOfFiles +=1
            if(ProbabilityOfHam <= ProbabilityOfSpam):
                InCorrectlyClassified+=1
    return InCorrectlyClassified,NumberOfFiles 

print("Naive Bayes on Test Set")

test_path = 'C:/Users/shash/Desktop/MS CE/Subject/Spring 2020/CS6375/CS-6375-Machine-Learning-master/Assignment 3/test'

HamTestPath = test_path + '\ham'
SpamTestPath = test_path + '\spam'

IncorrectlyClassifiedHam,TotalHamEmails = predicthamorSpam(HamTestPath, "ham")
IncorrectlyClassifiedSpam,TotalSpamEmails = predicthamorSpam(SpamTestPath,"spam")
AccuracyOfHamClassification = round(((TotalHamEmails - IncorrectlyClassifiedHam )/(TotalHamEmails ))*100,2)
AccuracyOfSpamClassification = round(((TotalSpamEmails -  IncorrectlyClassifiedSpam )/(TotalSpamEmails))*100,2)
AllEmailClassified = TotalHamEmails + TotalSpamEmails
TotalIncorrectClassified = IncorrectlyClassifiedHam + IncorrectlyClassifiedSpam
OverAllAccuracy = round(((AllEmailClassified  - TotalIncorrectClassified )/AllEmailClassified)*100,2)

print("Total number of Ham Emails: ", TotalHamEmails)
print("Total number of Spam Emails: ", TotalSpamEmails)

print("\nTest Files to be Classified is ", AllEmailClassified)
print("Number of Emails Classified as Ham: ", TotalHamEmails - IncorrectlyClassifiedHam)
print("Number of Emails Classified as Spam: ", TotalSpamEmails - IncorrectlyClassifiedSpam)
print("Naive Bayes Total accuracy for Test Emails: " + str(OverAllAccuracy) + "%")

print("\n")

print("Naive Bayes after removing stop words\n")
FindProbabilityOfWord("ham",1)
FindProbabilityOfWord("spam",1) 

IncorrectlyClassifiedHam,TotalHamEmails = predicthamorSpam(HamTestPath, "ham")
IncorrectlyClassifiedSpam,TotalSpamEmails = predicthamorSpam(SpamTestPath,"spam")
AccuracyOfHamClassification = round(((TotalHamEmails - IncorrectlyClassifiedHam )/(TotalHamEmails ))*100,2)
AccuracyOfSpamClassification = round(((TotalSpamEmails -  IncorrectlyClassifiedSpam )/(TotalSpamEmails))*100,2)
AllEmailClassified = TotalHamEmails + TotalSpamEmails
TotalIncorrectClassified = IncorrectlyClassifiedHam + IncorrectlyClassifiedSpam
OverAllAccuracy = round(((AllEmailClassified  - TotalIncorrectClassified )/AllEmailClassified)*100,2)

print("Total number of files: ", AllEmailClassified)
print("Number of Emails Classified as Ham: ", TotalHamEmails - IncorrectlyClassifiedHam)
print("Number of Emails Classified as Spam: ", TotalSpamEmails - IncorrectlyClassifiedSpam)
print("Naive Bayes Total accuracy for Test Emails: " + str(OverAllAccuracy) + "%")  
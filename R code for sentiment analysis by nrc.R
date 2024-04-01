install.packages("syuzhet")
install.packages("lubridate")
install.packages("ggplot2")
install.packages("scales")
install.packages("reshape2")
install.packages("dplyr")
install.packages("tm")
install.packages("devtools")
install.packages("wordcloud")
install.packages("NLP")
install.packages("RColorBrewer")
install.packages("SnowballC")
install.packages("topicmodels")
install.packages("data.table")
install.packages("stringi")
install.packages("qdap")
install.packages("plyr")
install.packages("gridExtra")

library("syuzhet")
library("lubridate")
library("ggplot2")
library("scales")
library("reshape2")
library("dplyr")
library("tm")
library("devtools")
library("wordcloud")
library("NLP")
library("RColorBrewer")
library("SnowballC")
library("topicmodels")
library("data.table")
library("stringi")
library("qdap")
library("plyr")
library("gridExtra")

##Actual code
tweets<- read.csv(file.choose(),header= TRUE)
head(tweets)
#documents corpus
tweetsCorpus<- Corpus(VectorSource(tweets$Status.text))
inspect(tweetsCorpus[3])
#Converting it into lower case
tweetsCorpus<- tm_map(tweetsCorpus, content_transformer(stri_trans_tolower))
#removing special character from the text
removeNumPunct<- function(x) gsub("[^[:alpha:][:space:]]*","",x)
tweetsCorpus<- tm_map(tweetsCorpus, content_transformer(removeNumPunct))
#removing stopwords
tweetsCorpus<- tm_map(tweetsCorpus,removeWords,stopwords("english"))
#removing whitespace
tweetsCorpus<-tm_map(tweetsCorpus, stripWhitespace)
#removing single words
singlewords<- function(x) gsub(" . "," ", x)
tweetsCorpus<- tm_map(tweetsCorpus, content_transformer(singlewords))
##By above stemming is done
sentimentscores<- get_nrc_sentiment(tweetsCorpus$content)
head(sentimentscores)
# Visualize
#Barplot
barplot(colSums((sentimentscores),las= 2, col= rainbow(10),ylab='count',main= 'barplot of Sentiments'))


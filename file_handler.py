# coding: utf-8
import csv, os, shutil, re
import nltk

def getGamesData(file_name, tokenizer=nltk.word_tokenize):
    remove_tag = re.compile('^[ ]+|[\n]+')
    replace_dot_tag = re.compile('[ ]*[.]+[ ]*')
    records = loadTxT(file_name)
    games = []
    corpus = []
    titles = []
    corpus_words = []
    words = []
    s_start = False
    for record in records:
        if record == '\n':
            if s_start :
                s_start = False
                sentences = replace_dot_tag.sub('.', remove_tag.sub('', sentences)).split('.')
                if sentences[-1] == '': sentences = sentences[:-1]
                game['sentences'] = sentences
                corpus += sentences
                for s in sentences:
                    tokens = tokenizer(s)
                    corpus_words.append(tokens)
                    words += tokens
                games.append(game)
            continue
        t_index = record.find('GameTitle : ')
        if t_index > -1:
            s_start = True
            title = record[t_index+len('GameTitle : '):].replace('\n','')
            game={'title':title}
            titles.append(title)
            sentences = ''
            continue
        elif s_start: sentences += record

    return {'games':games, 'corpus':corpus, 'titles':titles, 'corpus_words':corpus_words, 'words':words}

def getSentiCorpus(path):
    filter = re.compile('["/\'\[\],]+|[ ]+$|[\.]+|^[0-9]+$|…|'
                        '^[0-9]{2,4}[\-/.][0-9]{2}[\-/.][0-9]{2}( [0-9]{2}:[0-9]{2}(:[0-9]{2})?)?$|'
                        '[\(\{\[].*[\)\}\]]')
    files = os.listdir(path)
    sentences = []
    tokens = []
    polarity = []
    for f in files:
        lines = loadCSV(path+f, delimiter='\t')
        for line in lines:
            if (line[2] == 'SubjTag' or line[2] == 'ObjTag'):
                sentence = filter.sub('', line[13])
                if len(sentence) < 1: continue
                polarity.append(line[7])
                sentences.append(sentence)

                ts = line[14].split(' ')
                temp = []
                for t in ts:
                    s = filter.sub('', t[:t.rfind('/')])
                    if len(s) > 0: temp.append(s)
                tokens.append(temp)
    return {'sentences':sentences, 'tokens':tokens, 'polarity':polarity}

def getSentiDictionary():
    lines = loadCSV('sd.csv', 1)
    return {w:s for w, p, s in lines}

#########################################################################################################
######################################### TXT ###########################################################

def saveTxT(data, file_name):
    if ".txt" not in file_name: file_name = file_name+".txt"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    directory = path[:path.rfind('/')]
    if not os.path.isdir(directory):
        makeDirectories(directory)

    with open(path, "w", encoding='utf-8') as f:
        f.writelines(data)
    return True

def loadTxT(file_name, column_rows=0):
    if ".txt" not in file_name: file_name = file_name+".txt"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    if not os.path.isfile(path) :
        return None

    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    return lines[column_rows:]

#########################################################################################################
######################################### CSV ###########################################################

def saveCSV(data, file_name, column_sec=[]):
    if ".csv" not in file_name: file_name = file_name+".csv"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    directory = path[:path.rfind('/')]
    if not os.path.isdir(directory):
        makeDirectories(directory)

    csv_file = open(path, "w", newline='\n')
    cw = csv.writer(csv_file, delimiter=',', quotechar='|')

    if len(column_sec) > 0:
        for columns in column_sec:
            cw.writerow(columns)
    for row in data:
        cw.writerow(row)
    csv_file.close()
    return True

def loadCSV(file_name, column_rows=0, delimiter=',', quotechar='|'):
    if ".csv" not in file_name: file_name = file_name+".csv"

    if ":\\" in file_name or ":/" in file_name : path = file_name.replace('\\','/')
    else : path = getStoragePath()+file_name.replace('\\','/').replace(":", "")
    if not os.path.isfile(path) :
        return None
    csv_file = open(path, "r")
    cr = csv.reader(csv_file, delimiter=delimiter, quotechar=quotechar)

    lines = []
    for line in cr:
        lines.append(line)

    csv_file.close()

    return lines[column_rows:]

#########################################################################################################
#########################################################################################################
def makeDirectories(directory):
    if '\\' in directory: directory = directory.replace('\\', '/')
    if '/' in directory:
        u_dir = directory[:directory.rfind('/')]
        if not os.path.isdir(u_dir):
            makeDirectories(u_dir)
    if not os.path.isdir(directory):
        os.makedirs(directory)
def getStoragePath():
    StoragePath = os.getcwd().replace('\\', '/')+'/'
    return StoragePath

def clearCaches():
    cache_dir = 'C:/Users/Wooyoung/AppData/Local/Temp/'
    if not os.path.isdir(cache_dir) : return
    files = os.listdir(cache_dir)
    for file in files:
        if 'tmp' in file:
            if not os.path.isfile(cache_dir+file) :
                try: shutil.rmtree(cache_dir+file)
                except OSError as e: pass


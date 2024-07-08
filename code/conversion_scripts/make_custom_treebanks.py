'''
Split the UD treebanks into individual files for each unique work in the treebank.
'''
import re, glob, json, os
import random
from copy import deepcopy
from convert_ud_treebanks import get_sentences, SAVE_PATH

TREEBANKS_PATH = SAVE_PATH
LAS_DIR = TREEBANKS_PATH + 'lasla/'
UD_DIR = TREEBANKS_PATH + 'full_ud_sets/'

# ids: unique works
# treebanks: parallel to ids, the treebank each work is in
treebanks = ["perseus", "perseus", "perseus", "perseus", "proiel", "proiel", "proiel", "perseus", "perseus", "perseus", "perseus", "perseus", "perseus", "perseus", "proiel", "proiel", "llct", "ittb", "ittb", "ittb", "udante", "udante", "udante", "udante", "udante"]
ids = ["phi0474.phi013", "phi0690.phi003", "phi0620.phi001", "phi0631.phi001", "De officiis", "Commentarii belli Gallici", "Epistulae ad Atticum", "phi1221.phi007", "phi0959.phi006", "phi0972.phi001", "phi0975.phi001", "phi1348.abo012", "phi1351.phi005", "tlg0031.tlg027", "Jerome's Vulgate", "Opus agriculturae", "", "forma", "scg", "forma", "DVE", "Mon", "Epi", "Que", "Egl"]
new_ids = ["cicero_in-catilinam", "vergil_aeneid", "propertius_elegies", "sallust_bellum-catilinae", "cicero_de-officiis", "caesar_gallic-war", "cicero_letters-to-atticus", "augustus_res-gestae", "ovid_metamorphoses", "petronius_satyricon", "phaedrus_fabulae", "suetonius_life-of-augustus", "tacitus_historiae", "jerome_vulgata-Revelation-Pers", "jerome_vulgata", "palladius_opus-agriculturae", "llct_antiquiores", "aquinas_forma", "aquinas_summa-contra-gentiles", "aquinas_forma", "dante_de-vulgari-eloquentia", "dante_monarchia", "dante_letters", "dante_questio-de-aqua-et-terra", "dante_eclogues"]
time_periods = ["classical", "classical", "classical", "classical", "classical", "classical", "classical", "classical", "classical", "classical", "classical", "classical", "classical", "late", "late", "late", "late, medieval", "medieval", "medieval", "medieval", "medieval", "medieval", "medieval", "medieval", "medieval"]

def count_sents_per_work(treebanks_path=TREEBANKS_PATH):
    i = 0
    all_sent_counts = {}
    for treebank, id in zip(treebanks, ids):   
        print(f'Processing {id} in {treebank} treebank')
        
        new_id = new_ids[i]
        
        # dictionary so we can keep track of order of sentences and reorder later
        # key: sentence id, value: list of lines in the sentence
        sent_counts = {'train': 0, 'dev': 0, 'test': 0}

        for split in ['train', 'dev', 'test']:
            treebank_path = treebanks_path + f'MM-la_{treebank}-ud-{split}.conllu'
            with open(treebank_path, 'r') as f:
                lines = f.readlines()

            for j, line in enumerate(lines):
                # look for start of sentence
                if treebank == 'perseus' and line.startswith(f'# sent_id = {id}'):
                    sent_counts[split] += 1
                elif treebank == 'proiel' and line.startswith(f'# source = {id}'):
                    sent_counts[split] += 1
                elif treebank == 'llct' and line.startswith('# sent_id'):
                    sent_counts[split] += 1
                elif treebank == 'ittb' and line.startswith(f'# sent_id') and re.match(r'# reference = ittb-' + id, lines[j+2]):
                    sent_counts[split] += 1
                elif treebank == 'udante' and line.startswith(f'# sent_id = {id}'): 
                    sent_counts[split] += 1

        all_sent_counts[new_id] = sent_counts
        i += 1

    return all_sent_counts

def make_full_sets(treebanks_path=TREEBANKS_PATH, save_path=SAVE_PATH):
    if not os.path.exists(save_path + 'full_ud_sets/'):
        os.makedirs(save_path + 'full_ud_sets/')
    
    i = 0

    for treebank, id in zip(treebanks, ids):   
        print(f'Processing {id} in {treebank} treebank')
        
        # Create a new file for this work/id (it may appear in multiple train/dev/test splits)
        new_id = new_ids[i]
        save_file = save_path + f'full_ud_sets/{new_id}.conllu'
        
        # dictionary so we can keep track of order of sentences and reorder later
        # key: sentence id, value: list of lines in the sentence
        sents = {}

        for split in ['train', 'dev', 'test']:
            
            treebank_path = treebanks_path + f'ud/MM-la_{treebank}-ud-{split}.conllu'
            
            if not os.path.exists(treebank_path): continue
            with open(treebank_path, 'r') as f:
                lines = f.readlines()

            sent = []
            searching = True
            found = False
            curr_id = -1
            for j, line in enumerate(lines):
                if searching:
                    # look for start of sentence, and find the id (ordering)
                    if treebank == 'perseus' and line.startswith(f'# sent_id = {id}'):
                        curr_id = int(line.split('@')[1].strip())
                        searching = False
                        found = True
                    elif treebank == 'proiel' and line.startswith(f'# source = {id}'):
                        curr_id_line = lines[j-2]
                        curr_id = int(re.match(r'# sent_id = (\d+)', curr_id_line).group(1))
                        searching = False
                        found = True
                    elif treebank == 'llct' and line.startswith('# sent_id'):
                        # curr id is 2 lines ahead
                        curr_id_line = lines[j+2]
                        match_ = re.match(r"# reference = document_id='([0-9\:]+)'-span='(\d+)'", curr_id_line)
                        curr_doc_id = match_.group(1)
                        curr_span = match_.group(2)
                        curr_id = curr_doc_id + '-' + curr_span
                        searching = False
                        found = True
                    elif treebank == 'ittb' and line.startswith(f'# sent_id') and re.match(r'# reference = ittb-' + id, lines[j+2]):
                        # curr id is 2 lines ahead
                        curr_id_line = lines[j+2]
                        curr_id = curr_id_line.split('-')[-1].strip()
                        curr_id = int(curr_id[1:])
                        searching = False
                        found = True
                    elif treebank == 'udante' and line.startswith(f'# sent_id = {id}'): 
                        curr_id = int(line.split('-')[1].strip())
                        searching = False
                        found = True

                    if found:
                        # add the first line of the sentence
                        if treebank == 'proiel':
                            sent.extend(lines[j-2:j+1])
                        else:
                            sent.append(line)
                        
                else: # found
                # if we reach the end of a sentence, add it to the dictionary

                    if line == '\n':
                        sent.append(line)
                        sents[curr_id] = sent
                        sent = []
                        searching = True
                        found = False

                    else:
                        sent.append(line)
    

            # add the last sentence if it wasn't added
            if len(sent) > 0:
                sents[curr_id] = sent
        print(f'Found {len(sents)} sentences for {new_id}')
        # write the sentences in order to the new file
        #print(f'Writing {new_id} to {save_file}')
        if treebank == 'llct':
            # sort by document id and then span
            sents = {k: v for k, v in sorted(sents.items(), key=lambda item: (item[0].split('-')[0], int(item[0].split('-')[1])))}
        else:
            sents = {k: v for k, v in sorted(sents.items(), key=lambda item: item[0])}

        with open(save_file, 'w') as f:
            for sent in sents.values():
                for line in sent:
                    f.write(line)

        i += 1

def get_book(line):
    '''Helper for split_bible()'''
    _, book = line.split('Vulgate, ')
    book = book.strip()  
    book = re.search(r'^(.*?) \d+', book).group(1)
    return book

def split_bible():
    '''Splits Bible (from PROIEL) into individual books'''
    bible_path = SAVE_PATH + 'full_ud_sets/jerome_vulgata.conllu'

    with open(bible_path, 'r') as f:
        lines = f.readlines()

    books = set()
    for line in lines:
        if line.startswith('# source = '):
            book = get_book(line)
            books.add(book)

    sentences = get_sentences(lines)

    books_to_sents = {book: [] for book in books}
    for sent in sentences:
        book = get_book(sent[2]) # sent[2] is the source line
        books_to_sents[book].append(sent)

    for book, sents in books_to_sents.items():
        # replace spaces with -
        book = book.replace(' ', '-')

        with open(SAVE_PATH + f'full_ud_sets/jerome_vulgata-{book}.conllu', 'w') as f:
            for sent in sents:
                for line in sent:
                    f.write(line)

def split_llct():
    '''Splits LLCT into volumes'''
    llct_path = SAVE_PATH + 'full_ud_sets/llct_antiquiores.conllu'

    with open(llct_path, 'r') as f:
        lines = f.readlines()
    
    sentences = get_sentences(lines, use_sent_ids=True)
    volume_to_sents = {}
    for sent_id, sent in sentences.items():
        ref_line = sent[2]
        doc_id = re.search(r"document_id='([\d:]+)'-span='\d+'", ref_line).group(1)
        vol_id = doc_id.split(':')[0]

        if vol_id not in volume_to_sents:
            volume_to_sents[vol_id] = []
        volume_to_sents[vol_id].append(sent)

    for vol_id, sents in volume_to_sents.items():
        with open(SAVE_PATH + f'full_ud_sets/llct_antiquiores-{vol_id}.conllu', 'w') as f:
            for sent in sents:
                for line in sent:
                    f.write(line)

def split_cic_letters():
    cic_path = SAVE_PATH + 'full_ud_sets/cicero_letters-to-atticus.conllu'
    with open(cic_path, 'r') as f:
        lines = f.readlines()

    sentences = get_sentences(lines, use_sent_ids=True)
    book_to_sents = {}
    for sent_id, sent in sentences.items():
        src_line = sent[2]
        book = re.search(r'Book (\d+)', src_line).group(1)

        if book not in book_to_sents:
            book_to_sents[book] = []
        book_to_sents[book].append(sent)

    for book, sents in book_to_sents.items():
        with open(SAVE_PATH + f'full_ud_sets/cicero_letters-to-atticus-{book}.conllu', 'w') as f:
            for sent in sents:
                for line in sent:
                    f.write(line)

def make_train_test_splits():
    split_file = SAVE_PATH + 'split_ids.json'
    with open(split_file, 'r') as f:
        work_to_split_ids = json.load(f)

    if not os.path.exists(SAVE_PATH + 'train_test_splits/'):
        os.makedirs(SAVE_PATH + 'train_test_splits/')

    for work, splits in work_to_split_ids.items():
    
        # a lasla file
        if work[0].isupper():
            fnames = glob.glob(LAS_DIR + work + '*.conllup')
        else:
            fnames = [UD_DIR + work + '.conllu']

        # sent id -> sentence
        all_sents = {}
        for fname in fnames: # could be multiple files for a work
            with open(fname, 'r') as f:
                lines = f.readlines()
            sentences = get_sentences(lines, use_sent_ids=True)
            all_sents.update(sentences)

        for split_type, sent_ids in splits.items():
            split_sents = [all_sents[sent_id] for sent_id in sent_ids]
            with open(SAVE_PATH + 'train_test_splits/' + work + '-' + split_type + '.conllu', 'w') as f:
                for sent in split_sents:
                    for line in sent:
                        f.write(line)
                    # includes newlines already, so no need to add one

if __name__ == '__main__':
    print('Making full UD sets')
    make_full_sets() # separate files for each UD work
    print('-----------------')
    print('Splitting Bible, LLCT, and Cicero Letters')
    split_bible()
    split_llct()
    split_cic_letters()
    print('-----------------')
    print('Making train/test splits')
    make_train_test_splits()
    
    # print the number of sentences in each split
    for treebank_file in glob.glob(SAVE_PATH + 'train_test_splits/*.conllu'):
        with open(treebank_file, 'r') as f:
            lines = f.readlines()
        sents = get_sentences(lines)
        f_name = treebank_file.split('/')[-1].split('.')[0]

        print(f'{f_name}, {len(sents)}')
    
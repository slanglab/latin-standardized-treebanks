import re, sys, glob, os, random
from typing import List

MM_UD_TREEBANKS_PATH = '../../data/original_treebanks/Latin-variability/morpho_harmonization/morpho-harmonized-treebanks/'
LASLA_TREEBANK_PATH = '../../data/original_treebanks/LASLA/conllup/'
SAVE_PATH = '../../data/converted_treebanks/'
VERBOSE = True
TREEBANK_NAMES = ['perseus', 'proiel', 'ittb', 'llct', 'udante']

# columns in the converted conllu files, not the UD files
TAGS_TO_COLS = {
    'lemma': 2,
    'upos': 3,
    'xpos': 4,
    'morph': 5,
    'dep': 6,
    'head': 7,
    'original_morph': 8
}
MORPH_FEATS = ['Person', 'Number', 'Tense', 'Mood', 'Voice', 'Gender', 'Case', 'Degree']
MORPH_FEATS = sorted(MORPH_FEATS)
FEAT_TO_IDX = {MORPH_FEATS[i]: i for i in range(len(MORPH_FEATS))}

def convert_tags_per_treebank(tree_name: str, lines: List[str]) -> List[str]:
    '''
    tree_name: name of treebank, e.g. 'perseus'
    lines: lines from conllu file
    '''
    converted_lines = []
    prev_tok = ''
    next_tok = ''

    for i, line in enumerate(lines):

        if line.startswith('#') or len(line.rstrip()) == 0:
            converted_lines.append(line)
            continue
        elif re.match(r"\d+-\d+", line): # multiword token; skip these 
            continue
        
        cols = line.rstrip().split("\t")
        pos = cols[3]
        morph = cols[5]
        xpos = cols[4]
        traditional_info = cols[9]

        morph_list = ['' for _ in range(len(MORPH_FEATS))]

        # part of speech: not all treebanks have INTJ , 
        # so convert those to PART to be consistent
        if pos == 'INTJ': 
            pos = 'PART'
        #elif pos == '_':
        #    pos = 'X'

        # person
        person = 'None'
        match_ = re.search(r"Person=(\w+)", morph)
        if match_: person = match_.group(1)
        idx = FEAT_TO_IDX['Person']
        morph_list[idx] = 'Person=' + person

        # number
        number = 'None'
        match_ = re.search(r"Number=(\w+)", morph)
        if match_: 
            number = match_.group(1)
            if tree_name == 'lasla' and number == 'Plural':
                number = 'Plur'
        idx = FEAT_TO_IDX['Number']
        morph_list[idx] = 'Number=' + number

        # tense: use TraditionalTense field
        tense = 'None'
        # infinitives are present or perfect; check aspect
        
        # lasla rules
        # present: check for tense = Pres if finite; 
        #   voice=act and verbform=part if non-finite, aspect = imp
        # imperfect: aspect = imp, tense = Past
        # perfect: aspect = perf, tense = Past or None (not pqp or fut)
        # pluperfect: tense = Pqp
        # future: finite: tense = Fut, aspect != Perf, nonfinite: aspect = prosp
        # future perfect: tense = Fut, aspect = Perf

        # no tense if mood is ger, gdv, sup
           
        if tree_name == 'lasla':
            tense_match_ = re.search(r"Tense=(\w+)", morph) 
        else:
            tense_match_ = re.search(r"TraditionalTense=(\w+)", traditional_info)
        verbform_match = re.search(r"VerbForm=(\w+)", morph)
        aspect_match = re.search(r"Aspect=(\w+)", morph)

        if tense_match_: tense = tense_match_.group(1)
        if aspect_match: aspect = aspect_match.group(1)
        else: aspect = 'None'
        if verbform_match: verbform = verbform_match.group(1)
        else: verbform = 'None'
        
        if tense_match_ and tree_name != 'lasla': # UD treebanks
            if   tense == 'Praesens': tense = 'Pres'
            elif tense == 'Imperfectum': tense = 'Imp'
            elif tense == 'Perfectum': tense = 'Perf'
            elif tense == 'Plusquamperfectum': tense = 'Plup'
            elif tense == 'FuturumExactum': tense = 'FutP' # only in UDante
            elif tense == 'Futurum': 
                # need to check Aspect to see if Future or Future Perfect
                if aspect_match: 
                    aspect = aspect_match.group(1)
                    if aspect == 'Perf': tense = 'FutP'
                    else:                tense = 'Fut'
                else:
                    tense = 'None' # sometimes 'futurum' an adj has tense marked; ignore
        elif tree_name == 'lasla':
            # pres finites    OR   pres act participles/infinitives
            if tense == 'Pres' or (verbform in ['Part', 'Inf'] and aspect == 'Imp'):
                tense = 'Pres'
            # imperf finites only 
            elif tense == 'Past' and aspect == 'Imp':
                tense = 'Imp'
            # perf finites or perf pass participles; exclude fut perf and plurperf verbs
            elif aspect == 'Perf' and tense not in ['Pqp', 'Fut']: # tense is Past or None
                tense = 'Perf'
            # pluperfect
            elif tense == 'Pqp':
                tense = 'Plup'
            elif (tense == 'Fut' and aspect != 'Perf') or aspect == 'Prosp':
                tense = 'Fut'
            elif tense == 'Fut' and aspect == 'Perf':
                tense = 'FutP'
        idx = FEAT_TO_IDX['Tense']
        morph_list[idx] = 'Tense=' + tense

        # mood: use TraditionalMood field
        mood = 'None'
        if tree_name == 'lasla':
            mood_match_ = re.search(r"Mood=(\w+)", morph)
        else:
            mood_match_ = re.search(r"TraditionalMood=(\w+)", traditional_info)
        if mood_match_: mood = mood_match_.group(1)

        if mood_match_ and tree_name != 'lasla': # UD treebanks
            if   mood == 'Indicativus': mood = 'Ind'
            elif mood == 'Subiunctivus': mood = 'Sub'
            elif mood == 'Imperativus': mood = 'Imp'
            elif mood == 'Participium': mood = 'Part'
            elif mood == 'Gerundium': mood = 'Ger'
            elif mood == 'Gerundivum': mood = 'Gdv'
            elif mood == 'Sup': mood = 'Sup'
            elif mood == 'Infinitivus': mood = 'Inf'
        
        # check for infinitives in VerbForm field in UD
        elif tree_name != 'lasla' and verbform_match and verbform == 'Inf':
            mood = 'Inf'
            # check aspect to find out if present or perfect
            if aspect_match and aspect == 'Perf': 
                tense = 'Perf'
            elif aspect_match:
                tense = 'Pres'
            idx = FEAT_TO_IDX['Tense']
            morph_list[idx] = 'Tense=' + tense

        elif tree_name == 'lasla':
            # can take non-finite verbforms directly as mood
            if verbform_match and verbform in ['Part', 'Ger', 'Gdv', 'Sup', 'Inf']:
                mood = verbform
            
            # else verbform is finite, so take mood straight from morph,
            # OR not a verb, so mood is None (already set above)
            #elif mood_match_: mood = mood_match_.group(1)
  
        idx = FEAT_TO_IDX['Mood']
        morph_list[idx] = 'Mood=' + mood
        
        # fix tenses, numbers, based on mood
        # no number for gerunds, infinitives, and supines;
        # no tense for gerunds, supines, gerundives
        if mood in ['Ger', 'Inf', 'Sup']:
            idx = FEAT_TO_IDX['Number']
            morph_list[idx] = 'Number=None'
        if mood in ['Ger', 'Gdv', 'Sup']:
            idx = FEAT_TO_IDX['Tense']
            morph_list[idx] = 'Tense=None'
        # add no case for Inf?
 
        # voice
        voice = 'None'
        if pos == 'AUX': 
            voice = 'Act'  # if POS is AUX, voice is active
        elif mood == 'Ger':
            voice = 'Act'
        elif mood == 'Gdv':
            voice = 'Pass'
        elif mood == 'Sup':
            # check if prev or next tok is 'iri'
            if len(lines) > i+1 and not (lines[i+1].startswith('#') or len(lines[i+1].rstrip()) == 0):
                next_tok = lines[i+1].split('\t')[1]
            else:
                next_tok = ''
                
            if next_tok.lower() == 'iri' or prev_tok.lower() == 'iri':
                voice = 'Pass'
            else:
                voice = 'Act'
        else:
            match_ = re.search(r"Voice=(\w+)", morph)
            if match_: voice = match_.group(1)
        idx = FEAT_TO_IDX['Voice']
        morph_list[idx] = 'Voice=' + voice

        # gender
        gender = 'None'
        match_ = re.search("Gender=(\w+,?\w*?,?\w*?)\|", morph) # Gender=Fem,Masc,Neut is possible in LASLA
        if match_: gender = match_.group(1)
        if mood in ['Ger', 'Inf', 'Sup']:
            gender = 'None'
        idx = FEAT_TO_IDX['Gender']
        morph_list[idx] = 'Gender=' + gender

        # case
        case = 'None'
        match_ = re.search("Case=(\w+)", morph)
        if match_: case = match_.group(1)
        idx = FEAT_TO_IDX['Case']
        morph_list[idx] = 'Case=' + case

        # degree
        degree = 'None'
        match_ = re.search("Degree=(\w+)", morph)
        if match_: 
            degree = match_.group(1)
            if degree == 'Pos' or degree == 'Dim': degree = 'None' # ignore these, not all treebanks have them; and Pos is default
            elif degree == 'Sup' or degree == 'Abs': degree = 'Sup' # both are superlative  
            # else, degree must be 'Cmp'
        idx = FEAT_TO_IDX['Degree']
        morph_list[idx] = 'Degree=' + degree

        # dependency relation
        if tree_name == 'lasla':
            dep = '_'
            head = '_'
        else:
            head = cols[6]
            dep = cols[7]

        # remove any morph feats that are 'None'
        morph_list = [feat for feat in morph_list if feat.split('=')[1] != 'None']
        morph_str = '|'.join(morph_list)
        if morph_str == '': morph_str = '_' # if no morph feats, use '_'

        #        num      token    lemma                                    # keep original morphological info
        attrs = [cols[0], cols[1], cols[2], pos, xpos, morph_str, head, dep, '_', morph]
        new_line = '\t'.join(attrs).strip() + '\n'
        converted_lines.append(new_line)

        prev_tok = cols[1]

    return converted_lines

def convert_all_treebanks(save_path=SAVE_PATH):
    ud_save_path = save_path + 'ud/'
    for treebank_file in glob.glob(MM_UD_TREEBANKS_PATH + '*/*.conllu'):
        if VERBOSE: print('Processing', treebank_file)
        tree_name = re.search(r"la_(\w+)-ud-", treebank_file).group(1)
        
        with open(treebank_file, 'r') as f:
            lines = f.readlines()

        final_lines = convert_tags_per_treebank(tree_name, lines)

        # save to new file
        if not os.path.exists(ud_save_path):
            os.makedirs(ud_save_path)
        new_file = ud_save_path + treebank_file.split('/')[-1]
        with open(new_file, 'w') as f:
            for line in final_lines: 
                f.write(line)
        
    # lasla files
    lasla_save_path = save_path + 'lasla/'
    for treebank_file in glob.glob(LASLA_TREEBANK_PATH + '*.conllup'):
        if VERBOSE: print('Processing', treebank_file)
        tree_name = 'lasla'
        
        with open(treebank_file, 'r') as f:
            lines = f.readlines()
        
        final_lines = convert_tags_per_treebank(tree_name, lines)

        # save to new file
        if not os.path.exists(lasla_save_path):
            os.makedirs(lasla_save_path)
        new_file = lasla_save_path + treebank_file.split('/')[-1]
        with open(new_file, 'w') as f:
            for line in final_lines: 
                f.write(line)
                

def get_sentences(lines: List[str], use_sent_ids=False):
    '''
    lines: lines from a conllu file
    '''
    sents = []
    sent = []
    for line in lines:
        sent.append(line)
        
        if len(line.rstrip()) == 0:
            sents.append(sent)
            sent = []

    # add last sentence if not empty
    if len(sent) > 0:
        sents.append(sent)

    # return a dict of {sent_id: sent}
    if use_sent_ids:
        sent_dict = {}
        for sent in sents:
            sent_id = None

            # should be in first 3 lines
            for line in sent:
                if line.startswith('# sent_id'):
                    sent_id = line.split('sent_id =')[1].strip()
                    break

            sent_dict[sent_id] = sent
        return sent_dict
    
    # else, return list of sentences, each a list of lines (strings)
    else:
        return sents
     

if __name__ == "__main__":
    convert_all_treebanks()

    

    
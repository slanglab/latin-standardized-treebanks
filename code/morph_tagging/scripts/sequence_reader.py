import os
from generate_tagset import extract_feats_from_line

def read_tagset(filename):
	tags={}
	curr_label_type=None
	with open(filename) as file:
		for line in file:
			if line.startswith("#"):
				label_type=line[2:].rstrip()
				tags[label_type]={}
				curr_label_type=label_type
			elif line == '\n': 
				continue
			else:
				cols=line.rstrip().split("\t")
				tags[curr_label_type][cols[0]]=int(cols[1])
	return tags


def read_filenames(filename):
	inpaths=[]
	outpaths=[]
	with open(filename) as file:
		for line in file:
			cols=line.rstrip().split("\t")
			
			if len(cols) == 2:
				inpaths.append(cols[0])
				outpaths.append(cols[1])

	return inpaths, outpaths


def read_annotations(filename, tagset, labeled):

	""" Read tsv data and return sentences and [word, {feat/upos: val}, sentenceID, filename] list 
		-- labeled: whether labels are present in the data (gold file) or not (test file)
	"""

	with open(filename, encoding="utf-8") as f:
		sentence = []
		sentence.append(["[CLS]", -100, -1, -1, None])
		sentences = []
		sentenceID=0
		for line in f:
			if line.startswith("#"):
				continue

			if len(line) > 0:
				if line == '\n':
					sentenceID+=1

					sentence.append(["[SEP]", -100, -1, -1, None])
					sentences.append(sentence)
					sentence = []
					sentence.append(["[CLS]", -100, -1, -1, None])

				else:
					data=[]
					split_line = line.rstrip().split("\t")
					

					# LOWERCASE for latin

					word=split_line[1].lower()
					label=0
					if labeled:
						label = extract_feats_from_line(line, tagset=tagset)

					data.append(word)
					data.append(label)
					
					data.append(sentenceID)
					data.append(filename)

					sentence.append(data)
		
		sentence.append(["[SEP]", -100, -1, -1, None])		
		if len(sentence) > 2:
			sentences.append(sentence)

	return sentences

def prepare_annotations_from_file(filename, tagset, labeled=True):
	""" Read a single file of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
	"""

	sentences = []
	annotations = read_annotations(filename, tagset, labeled)
	sentences += annotations
	return sentences

def prepare_annotations_from_folder(folder, tagset, labeled=True):

	""" Read folder of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
	"""

	sentences = []
	for filename in os.listdir(folder):
		annotations = read_annotations(os.path.join(folder,filename), tagset, labeled)
		sentences += annotations
	print("num sentences: %s" % len(sentences))
	return sentences

def prepare_annotations_from_files(filenames, tagset, labeled=True):

	""" Read multiple files of annotations, returning:
		-- a list of sentences, each a list of [word, label, sentenceID, filename]
	"""

	sentences = []
	for filename in filenames:
		annotations = read_annotations(filename, tagset, labeled)
		sentences += annotations
	print("num sentences: %s" % len(sentences))
	return sentences
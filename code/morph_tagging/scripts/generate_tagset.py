import sys, re

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

# column in training data that contains the tag type
def extract_feats_from_line(line, tagset=None):
	'''Feats, including UPOS'''
	cols = line.rstrip().split("\t")
	pos_col = TAGS_TO_COLS['upos']
	morph_col = TAGS_TO_COLS['morph']
	original_morph_col = TAGS_TO_COLS['original_morph']

	upos = cols[pos_col]
	morph_str = cols[morph_col]
	original_morph_str = cols[original_morph_col]
	#print(original_morph_str)

	feats_to_vals = {}
	if morph_str != "_":
		morph_list = morph_str.split("|")
		feats_to_vals = {morph.split('=')[0]: morph.split('=')[1] for morph in morph_list}
	feats_to_vals['upos'] = upos

	for feat in MORPH_FEATS:
		if feat not in feats_to_vals:
			feats_to_vals[feat] = "None"

	# special cases
	# we won't consider LASLA gender if it has multiple values
	if ',' in feats_to_vals['Gender']:
		feats_to_vals.pop('Gender')

	# lasla doesn't annotate "Person" for pronouns,
	# so we'll ignore these annotations 
	if feats_to_vals['Person'] == 'None' and re.search('PronType=Prs', original_morph_str):
		feats_to_vals.pop('Person')

	# if given tagset, convert values from str to int
	if tagset is not None:
		for feat, val in feats_to_vals.items():
			feats_to_vals[feat] = tagset[feat][val]
	
	return feats_to_vals

def proc(filenames: list, do_combined: bool, outdir: str):
	# dictionary of {label type: {label: 1}} (to get unique labels)
	# e.g. {"pos": {"NOUN": 1, "VERB": 1}, "voice": {"pass": 1, "act": 1}}
	if do_combined:
		tags = {"combined": set()}
	else:
		tags = {k: set() for k in FEAT_TO_IDX.keys()}
		tags['upos'] = set()

	for filename in filenames:
		with open(filename) as file:
			for line in file:
				if line.startswith("#") or len(line.rstrip()) == 0:
					continue

				feats_to_vals = extract_feats_from_line(line)
				
				if do_combined: # combine labels into one
					label = ""
					for feat in MORPH_FEATS:
						label += feats_to_vals[feat] + "_"
					label = label[:-1] # remove last underscore
					tags['combined'].add(label)
				
				else:
					for feat, val in feats_to_vals.items():
						tags[feat].add(val)


	with open(outdir + "morph.tagset", "w") as outfile:
		for label_type, labels in tags.items():
			outfile.write("# %s\n" % label_type)
			for idx, tag in enumerate(labels):
				outfile.write("%s\t%d\n" % (tag, idx))
			outfile.write("\n")

if __name__ == "__main__":
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument("-c", "--combined_labels", help="whether to combine labels", action="store_true", default=False)
	parser.add_argument("-o", "--outdir", help="output directory", default="../converted_treebanks/", type=str)
	parser.add_argument("-f", "--filenames", help="filenames", required=True, nargs="+")

	args = parser.parse_args()

	do_combined = args.combined_labels
	proc(args.filenames, do_combined, args.outdir)



#proc(sys.argv[1:])
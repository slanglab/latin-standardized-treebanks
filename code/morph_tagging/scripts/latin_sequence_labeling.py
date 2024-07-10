"""
Sequence labeling with BERT + supervised fine-tuning


"""

import os,sys,argparse
from tqdm import tqdm
from transformers import BertModel, BertPreTrainedModel
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import sequence_reader, sequence_eval
import numpy as np
from tensor2tensor.data_generators import text_encoder
from copy import deepcopy
from generate_tagset import TAGS_TO_COLS, MORPH_FEATS, FEAT_TO_IDX
IGNORE_LABEL = -100

torch.manual_seed(0)
np.random.seed(0)

#batch_size=32
TOTAL_BATCH_SIZE = 32
batch_size=8
dropout_rate=0.25
bert_dim=768

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('')
print("********************************************")
print("Running on: {}".format(device))
print("********************************************")
print('')


class LatinTokenizer():
  def __init__(self, encoder):
    self.vocab={}
    self.reverseVocab={}
    self.encoder=encoder

    self.vocab["[PAD]"]=0
    self.vocab["[UNK]"]=1
    self.vocab["[CLS]"]=2
    self.vocab["[SEP]"]=3
    self.vocab["[MASK]"]=4
    

    for key in self.encoder._subtoken_string_to_id:
      self.vocab[key]=self.encoder._subtoken_string_to_id[key]+5
      self.reverseVocab[self.encoder._subtoken_string_to_id[key]+5]=key


  def convert_tokens_to_ids(self, tokens):
    wp_tokens=[]
    for token in tokens:
      if token == "[PAD]":
        wp_tokens.append(0)
      elif token == "[UNK]":
        wp_tokens.append(1)
      elif token == "[CLS]":
        wp_tokens.append(2)
      elif token == "[SEP]":
        wp_tokens.append(3)
      elif token == "[MASK]":
        wp_tokens.append(4)

      else:
        wp_tokens.append(self.vocab[token])
    return wp_tokens

  def tokenize(self, text):
    tokens=text.split(" ")
    wp_tokens=[]
    for token in tokens:
      if token in {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}:
      	wp_tokens.append(token)
      else:
        wp_toks=self.encoder.encode(token)

        for wp in wp_toks:
          wp_tokens.append(self.reverseVocab[wp+5])
    return wp_tokens


class BertForSequenceLabeling(nn.Module):

	def __init__(self, tagset, tokenizerPath=None, bertPath=None, freeze_bert=False):
		super(BertForSequenceLabeling, self).__init__()

		encoder = text_encoder.SubwordTextEncoder(tokenizerPath)
		# encoder = tfds.features.text.SubwordTextEncoder.load_from_file(vocab_filename)

		self.tokenizer = LatinTokenizer(encoder)

		self.num_labels_per_tag = {}
		self.classifiers = {}
		for label_type, labels in tagset.items():
			n_labels = len(labels)
			self.num_labels_per_tag[label_type] = n_labels
			self.classifiers[label_type] = nn.Linear(bert_dim, n_labels)
		self.classifiers = nn.ModuleDict(self.classifiers)

		self.bert = BertModel.from_pretrained(bertPath)

		self.bert.eval()
		
		if freeze_bert:
			for param in self.bert.parameters():
				param.requires_grad = False

		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, transforms=None, labels=None):
		'''
		return dict of losses, one for each prediction head
		'''
		input_ids = input_ids.to(device)
		attention_mask = attention_mask.to(device)
		#transforms = transforms.to(device)
		#labels = labels.to(device)
		#print("transforms:", transforms)
		#print("labels:", labels)
		if labels is not None:
			loss_fct = CrossEntropyLoss(ignore_index=-100)
		
		sequence_outputs, pooled_outputs = self.bert(input_ids, token_type_ids=None, attention_mask=attention_mask)
		all_layers=sequence_outputs
		#out=torch.matmul(transforms,all_layers)

		losses = {}
		for label_type in self.num_labels_per_tag.keys():
			this_transforms = transforms[label_type]
			this_transforms = this_transforms.to(device)
			
			out = torch.matmul(this_transforms, all_layers)
			
			logits = self.classifiers[label_type](out)
			num_labels = self.num_labels_per_tag[label_type]

			if labels is not None:
				this_labels = labels[label_type]
				this_labels = this_labels.to(device)
				loss = loss_fct(logits.view(-1, num_labels), this_labels.view(-1))
				losses[label_type] = loss
			else:
				losses[label_type] = logits

		return losses

	def predict(self, dev_file, tagset, outfile,  output_probs=False):

		rev_tagset = {k1: {v2: k2 for k2, v2 in v1.items()} for k1, v1 in tagset.items()}

		dev_orig_sentences = sequence_reader.prepare_annotations_from_file(dev_file, tagset, labeled=False)
		dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering=model.get_batches(dev_orig_sentences, batch_size)

		model.eval()

		bcount=0

		with torch.no_grad():

			#ordered_preds=[]
			ordered_preds_dict = {k: [] for k in self.num_labels_per_tag.keys()}

			#all_preds=[]
			#all_golds=[]

			for b in range(len(dev_batched_data)):
				all_logits = self.forward(dev_batched_data[b], token_type_ids=None, attention_mask=dev_batched_mask[b], transforms=dev_batched_transforms[b])
				
				for label_type in self.num_labels_per_tag.keys():
					size = dev_batched_labels[b][label_type].shape
					num_labels = self.num_labels_per_tag[label_type]
					b_size = size[0]
					b_size_labels = size[1]
					#b_size_orig = size[2]

					logits = all_logits[label_type]
					logits = logits.view(-1, b_size_labels, num_labels)
					logits = logits.cpu()

					preds = np.argmax(logits, axis=2)
					for row in range(b_size):
						ordered_preds_dict[label_type].append([np.array(r) for r in preds[row]])
			
			preds_in_order = [None for i in range(len(dev_orig_sentences))]
			for i, ind in enumerate(dev_ordering):
				preds_in_order[ind] = {}
				for label_type in self.num_labels_per_tag.keys():
					preds_in_order[ind][label_type] = ordered_preds_dict[label_type][i]
				#preds_in_order[ind] = ordered_preds[i]
			
			with open(outfile, "w", encoding="utf-8") as out:
				for idx, sentence in enumerate(dev_orig_sentences):

					# skip [CLS] and [SEP] tokens
					for t_idx in range(1, len(sentence)-1):
						sent_list=sentence[t_idx]
						token=sent_list[0]
						s_idx=sent_list[2]
						filename=sent_list[3]

						pred_feats_to_vals = {label_type: preds_in_order[idx][label_type][t_idx] for label_type in self.num_labels_per_tag.keys()}
						# get string representation of prediction
						pred_feats_to_vals = {label_type: rev_tagset[label_type][int(pred)] for label_type, pred in pred_feats_to_vals.items()}
						pos = pred_feats_to_vals.get("upos", "_")
						
						sorted_pred_feats = {feat: pred_feats_to_vals[feat] for feat in MORPH_FEATS}
						
						# remove any "None" values
						pred_feats_to_vals = {k: v for k, v in sorted_pred_feats.items() if v != "None"}
						morph_str = "|".join([f"{k}={v}" for k, v in pred_feats_to_vals.items()])

						if not pred_feats_to_vals:
							morph_str = "_"

						#         idx       tok    lemma  upos  xpos  feats    deps    head    misc
						out.write(f"{t_idx}\t{token}\t_\t{pos}\t_\t{morph_str}\t_\t_\t_\n")

						'''
						pred_list = [preds_in_order[idx][label_type][t_idx] for label_type in self.num_labels_per_tag.keys()]
						# get string representation of prediction
						pred_list = [rev_tagset[label_type][int(pred)] for label_type, pred in zip(self.num_labels_per_tag.keys(), pred_list)]
						pred = "_".join(pred_list)

						out.write("%s\t%s\n" % (token, pred))
						'''
					# longer than just "[CLS] [SEP]"
					if len(sentence) > 2:
						out.write("\n")

	def evaluate(self, dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, metric, tagset):
		model.eval()

		with torch.no_grad():

			ordered_preds={k: [] for k in self.num_labels_per_tag.keys()}

			all_preds={k: [] for k in self.num_labels_per_tag.keys()}
			all_golds={k: [] for k in self.num_labels_per_tag.keys()}
			losses = {k: 0.0 for k in self.num_labels_per_tag.keys()}

			for b in range(len(dev_batched_data)):
				all_logits = self.forward(dev_batched_data[b], token_type_ids=None, attention_mask=dev_batched_mask[b], transforms=dev_batched_transforms[b])

				for label_type in self.num_labels_per_tag.keys():
					logits = all_logits[label_type]
					logits=logits.cpu()

					ordered_preds[label_type] += [np.array(r) for r in logits]
					size=dev_batched_labels[b][label_type].shape

					num_labels = self.num_labels_per_tag[label_type]
					logits=logits.view(-1, size[1], num_labels)

					# if metric is CrossEntropyLoss
					if isinstance(metric, CrossEntropyLoss):
						loss = metric(logits.view(-1, num_labels), dev_batched_labels[b][label_type].view(-1))
						losses[label_type] += loss.item()
					else:
						for row in range(size[0]):
							for col in range(size[1]):
								if dev_batched_labels[b][label_type][row][col] != -100:
									pred=np.argmax(logits[row][col])
									all_preds[label_type].append(pred.cpu().numpy())
									all_golds[label_type].append(dev_batched_labels[b][label_type][row][col].cpu().numpy())
			
			if not isinstance(metric, CrossEntropyLoss):
				metric_dict = {}
				for label_type in self.num_labels_per_tag.keys():
					metric_dict[label_type] = metric(all_golds[label_type], all_preds[label_type], tagset[label_type])
				return metric_dict
			else:
				return losses

			#return metric(all_golds, all_preds, tagset)

	def get_batches(self, sentences, max_batch, labeled=True):

		maxLen=0
		for sentence in sentences:
			length=0
			for word in sentence:
				toks=self.tokenizer.tokenize(word[0])
				length+=len(toks)

			if length> maxLen:
				maxLen=length

		all_data=[]
		all_masks=[]
		all_transforms=[]
		all_labels = [] # for each sentence

		for sentence in sentences:
			tok_ids=[]
			input_mask=[]
			transform=[]
			labels = {k: [] for k in self.num_labels_per_tag.keys()}

			all_toks=[]
			n=0
			for idx, word in enumerate(sentence):
				toks=self.tokenizer.tokenize(word[0])
				all_toks.append(toks)
				n+=len(toks)

			cur=0
			for idx, word in enumerate(sentence):
				toks=all_toks[idx]
				ind=list(np.zeros(n))
				for j in range(cur,cur+len(toks)):
					ind[j]=1./len(toks)
				cur+=len(toks)
				transform.append(ind)

				tok_ids.extend(self.tokenizer.convert_tokens_to_ids(toks))

				input_mask.extend(np.ones(len(toks)))
				#print('word:', word)
				if type(word[1]) == int: # a CLS token; no label
					for label_type in self.num_labels_per_tag.keys():
						labels[label_type].append(word[1])
				else:
					'''
					split_label = word[1].split("_")
					for label_type, label_col in TAGS_TO_COLS.items():
						label_idx = label_col - 2 # subtract 2 because first two columns are # and word
						label = int(split_label[label_idx])
						labels[label_type].append(label)
					'''
					feats_to_vals = word[1]
					for feat in self.num_labels_per_tag.keys():
						if feat not in feats_to_vals:
							labels[feat].append(IGNORE_LABEL)
						else:
							labels[feat].append(feats_to_vals[feat])

			all_data.append(tok_ids)
			all_masks.append(input_mask)
			all_transforms.append(transform)
			all_labels.append(labels)

		lengths = np.array([len(l) for l in all_data])

		# Note sequence must be ordered from shortest to longest so current_batch will work
		ordering = np.argsort(lengths)
		
		ordered_data = [None for i in range(len(all_data))]
		ordered_masks = [None for i in range(len(all_data))]
		ordered_transforms = [None for i in range(len(all_data))]
		ordered_labels = [None for i in range(len(all_data))]

		for i, ind in enumerate(ordering):
			ordered_data[i] = all_data[ind]
			ordered_masks[i] = all_masks[ind]
			ordered_transforms[i] = all_transforms[ind]
			ordered_labels[i] = all_labels[ind]

		batched_data=[]
		batched_mask=[]
		batched_labels=[]
		batched_transforms=[]

		i=0
		current_batch=max_batch

		while i < len(ordered_data):
			#print(f'batch: {i} to {i+current_batch}')
			batch_data=ordered_data[i:i+current_batch]
			batch_mask=ordered_masks[i:i+current_batch]
			#batch_labels=ordered_labels[i:i+current_batch]
			batch_labels = {}
			batch_transforms = {}
			#print('ordered_labels:', ordered_labels)
			batch_transforms__=ordered_transforms[i:i+current_batch]
			for label_type in self.num_labels_per_tag.keys():
				batch_labels[label_type] = [sent[label_type] for sent in ordered_labels[i:i+current_batch]]
				batch_transforms[label_type] = deepcopy(batch_transforms__)

			max_len = max([len(sent) for sent in batch_data])
			#print('max_len:', max_len)
			max_labels = {}
			for label_type, labels in batch_labels.items():
				max_labels[label_type] = max([len(label) for label in labels])

			#print('len batch_data:', len(batch_data))
			#print('len batch_data[j]:', end='')
			for j in range(len(batch_data)): # for each sentence in the batch
				blen=len(batch_data[j])
				#print(f'  batch_data[{j}]')
				
				# pad the sentence
				for k in range(blen, max_len):
					batch_data[j].append(0)
					batch_mask[j].append(0)
					for label_type, labels in batch_labels.items():
						for z in range(len(batch_transforms[label_type][j])):
							batch_transforms[label_type][j][z].append(0)

				# pad the labels
				for label_type, labels in batch_labels.items():
					blab=len(batch_labels[label_type][j])
					for k in range(blab, max_labels[label_type]):
						batch_labels[label_type][j].append(-100)

				for label_type, labels in batch_labels.items():
					for k in range(len(batch_transforms[label_type][j]), max_labels[label_type]):
						batch_transforms[label_type][j].append(np.zeros(max_len))

			batched_data.append(torch.LongTensor(batch_data))
			batched_mask.append(torch.FloatTensor(batch_mask))
			for label_type in batch_labels.keys():
				#print('label_type:', label_type)
				#print('batch_transforms[label_type]:', batch_transforms[label_type])
				#print('batch_labels[label_type]:', batch_labels[label_type])
				#print()
				batch_labels[label_type] = torch.LongTensor(batch_labels[label_type])
				batch_transforms[label_type] = torch.FloatTensor(batch_transforms[label_type])
				
				
					
			batched_labels.append(batch_labels)
			batched_transforms.append(batch_transforms)
			#print('---------------------')
			#bsize=torch.FloatTensor(batch_transforms).shape
			
			i+=current_batch

			# adjust batch size; sentences are ordered from shortest to longest so decrease as they get longer
			#if max_len > 100:
			#	current_batch=12
			#if max_len > 200:
			#	current_batch=6

		return batched_data, batched_mask, batched_labels, batched_transforms, ordering


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-m','--mode', help='{train,test,predict,predictBatch}', required=True)
	parser.add_argument('--bertPath', help='path to pre-trained BERT', required=True)
	parser.add_argument('--tokenizerPath', help='path to Latin WordPiece tokenizer', required=True)
	
	parser.add_argument('-b','--batch_prediction_file', help='Filename containing input paths to tag, paired with output paths to write to', required=False)
	
	parser.add_argument('-i','--input_prediction_files', help='Filenames to tag', required=False, nargs='+')
	parser.add_argument('-o','--output_prediction_files', help='Filenames to write tagged text to', required=False, nargs='+')

	parser.add_argument('-r','--trainFiles', help='File containing training data', required=False, nargs='+')
	parser.add_argument('-e','--testFiles', help='File containing test data', required=False, nargs='+')
	parser.add_argument('-d','--devFiles', help='File containing dev data', required=False, nargs='+')

	parser.add_argument('-g','--tagFile', help='File listing tags + tag IDs.', required=False)

	parser.add_argument('-f','--modelFile', help='File to write model to/read from', required=False)
	parser.add_argument('-z','--metric', help='{accuracy,fscore,span_fscore}', required=False)
	parser.add_argument('--max_epochs', help='max number of epochs', required=False)
	parser.add_argument('--save_every_n', help='save model every n epochs; -1 to not save', required=False, default=-1, type=int)
	parser.add_argument('--load_from_checkpoint_num', help='load model from this epoch checkpoint; -1 if not loading from checkpoint', required=False, default=-1, type=int)

	args = vars(parser.parse_args())

	print(args)

	mode=args["mode"]
	
	tagFile=args["tagFile"]
	tagset=sequence_reader.read_tagset(tagFile)

	modelFile=args["modelFile"]
	max_epochs=args["max_epochs"]

	bertPath=args["bertPath"]
	tokenizerPath=args["tokenizerPath"]

	model = BertForSequenceLabeling(tagset, tokenizerPath=tokenizerPath, bertPath=bertPath, freeze_bert=False)

	model.to(device)

	if mode == "train":

		train_files=args["trainFiles"]
		dev_files=args["devFiles"]
	
		metric=None
		if args["metric"].lower() == "fscore":
			metric=sequence_eval.check_f1_two_lists
		elif args["metric"].lower() == "accuracy":
			metric=sequence_eval.get_accuracy
		elif args["metric"].lower() == "span_fscore":
			metric=sequence_eval.check_span_f1_two_lists

		# create folder for checkpoints
		model_dir = os.path.dirname(modelFile)
		model_name = os.path.basename(modelFile)
		if '.' in model_name:
			model_name, _ = model_name.split('.')
		checkpoint_dir = os.path.join(model_dir, model_name + '_checkpoints')
		if not os.path.exists(checkpoint_dir) and args["load_from_checkpoint_num"] != -1: os.makedirs(checkpoint_dir)

		# get training sentences
		trainSentences = sequence_reader.prepare_annotations_from_files(train_files, tagset)
		batched_data, batched_mask, batched_labels, batched_transforms, ordering=model.get_batches(trainSentences, batch_size)
		
		if dev_files is not None:
			devSentences = sequence_reader.prepare_annotations_from_files(dev_files, tagset)
			dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, dev_ordering=model.get_batches(devSentences, batch_size)

		learning_rate_fine_tuning=0.00005
		optimizer = optim.Adam(model.parameters(), lr=learning_rate_fine_tuning)
		
		maxScore=0
		best_val_loss=100000
		best_idx=0
		patience=10

		epochs=100
		if max_epochs is not None:
			epochs=int(max_epochs)

		if args["load_from_checkpoint_num"] != -1:
			start_epoch = args["load_from_checkpoint_num"]
			model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{model_name}_epoch{start_epoch}.pt'), map_location=device))
			optimizer.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{model_name}_epoch{start_epoch}_optimizer.pt')))
		else:
			start_epoch = 0
		
		grad_accum = int(TOTAL_BATCH_SIZE / batch_size)
		step_cnt = 0
		for epoch in range(start_epoch, epochs):
			model.train()
			big_loss = 0.0
			big_losses = {k: 0.0 for k in model.num_labels_per_tag.keys()}

			for b in range(len(batched_data)):
				if b % 10 == 0:
					print(b)
					sys.stdout.flush()
				
				losses = model(batched_data[b], token_type_ids=None, attention_mask=batched_mask[b], transforms=batched_transforms[b], labels=batched_labels[b])
				loss = sum(losses.values())
				big_loss += loss.item()
				big_losses = {k: big_losses[k] + v.item() for k, v in losses.items()}
				loss.backward()
				
				if step_cnt % grad_accum == 0:
					optimizer.step()
					model.zero_grad()
				step_cnt += 1
			
			print("loss: ", big_loss)
			print("losses: ", big_losses)
			sys.stdout.flush()

			score = 0.0
			val_loss = 0.0

			if dev_files is not None:
				print("\n***EVAL***\n")

				scores=model.evaluate(dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, metric, tagset)
				score = sum(scores.values())
				print('Scores:', scores)
				print("Score: %s" % score)
				
				val_losses = model.evaluate(dev_batched_data, dev_batched_mask, dev_batched_labels, dev_batched_transforms, CrossEntropyLoss(ignore_index=-100), tagset)
				val_loss = sum(val_losses.values())
				print('Val losses:', val_losses)
				print("Val loss: %s" % val_loss)

			sys.stdout.flush()
			if dev_files is None or (val_loss < best_val_loss and epoch > 0):
				torch.save(model.state_dict(), args["modelFile"])
				#torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch}_optimizer.pt'))
				if score > maxScore: maxScore = score
				best_idx=epoch
				best_val_loss = val_loss
			
			elif args["save_every_n"] != -1 and epoch % args["save_every_n"] == 0:
				torch.save(model.state_dict(), os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch}.pt'))
				torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, f'{model_name}_epoch{epoch}_optimizer.pt'))

			if epoch-best_idx > patience:
				print ("Stopping training at epoch %s" % epoch)
				break

		print("Best epoch: %s" % best_idx)
		print("Best score: %s" % maxScore)
		print("Best val loss: %s" % best_val_loss)


	elif mode == "test":

		metric=None
		if args["metric"].lower() == "fscore":
			metric=sequence_eval.check_f1_two_lists
		elif args["metric"].lower() == "accuracy":
			metric=sequence_eval.get_accuracy
		elif args["metric"].lower() == "span_fscore":
			metric=sequence_eval.check_span_f1_two_lists

		test_files=args["testFiles"]

		testSentences = sequence_reader.prepare_annotations_from_files(test_files, tagset)
		test_batched_data, test_batched_mask, test_batched_labels, test_batched_transforms, test_ordering=model.get_batches(testSentences, batch_size)

		model.load_state_dict(torch.load(modelFile, map_location=device))
		scores=model.evaluate(test_batched_data, test_batched_mask, test_batched_labels, test_batched_transforms, metric, tagset)
		score = sum(scores.values())
		print(scores)
		print("Final score: %s" % score)

	elif mode == "predict":

		# should be parallel
		predictFiles=args["input_prediction_files"]
		outFiles=args["output_prediction_files"]

		model.load_state_dict(torch.load(modelFile, map_location=device))
		 
		n = len(predictFiles)
		for i in range(n):
			pred_file = predictFiles[i]
			print(f'({i+1}/{n}) {pred_file}')
			pred_filename = os.path.basename(pred_file)
			if os.path.isfile(pred_file):
				out_file = outFiles[i] #+ pred_filename.split(".")[0] + "_pred.conllu" 
				model.predict(pred_file, tagset, out_file)


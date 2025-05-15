import os, random

fs = os.listdir('./extracted')

window = 5
chk_context = int(window / 2)

data_hasquery = []
data_none = []
for f in fs:
  txts = open('./extracted/{}'.format(f)).readlines()
  for i in range(len(txts)):
    if txts[i].strip().split('\t')[1] != 'null' and i - chk_context > 0 and i + chk_context < len(txts):
      cands = txts[i-chk_context:i+chk_context]
      tmp_utt = []
      tmp_qr = []
      for cand in cands:
        tmp_utt.append(cand.strip().split('\t')[0])
        if cand.strip().split('\t')[1] == 'null':
          continue
        tmp_qr.append(cand.strip().split('\t')[1])
      tmp_qr = list(set(tmp_qr))
      data_hasquery.append('{}\t{}'.format(' '.join(tmp_utt), ' '.join(tmp_qr)))
    elif txts[i].strip().split('\t')[1] == 'null' and i - chk_context > 0 and i + chk_context < len(txts):
      cands = txts[i-chk_context:i+chk_context]
      tmp_utt = []
      tmp_qr = []
      chk_null = False
      for cand in cands:
        if cand.strip().split('\t')[1] != 'null':
          chk_null = True
          break
        tmp_utt.append(cand.strip().split('\t')[0])
        tmp_qr.append(cand.strip().split('\t')[1])
      if chk_null:
        continue
      tmp_qr = list(set(tmp_qr))
      data_none.append('{}\t{}'.format(' '.join(tmp_utt), ' '.join(tmp_qr)))
      
random.shuffle(data_hasquery)
random.shuffle(data_none)
len_q = len(data_hasquery)
len_n = len_q * 4

train = data_hasquery[:int(len_q*0.8)] + data_none[:int(len_n*0.8)]
valid = data_hasquery[int(len_q*0.8):int(len_q*0.9)] + data_none[int(len_n*0.8):int(len_n*0.9)]
test = data_hasquery[int(len_q*0.9):] + data_none[int(len_n*0.9):int(len_n)]

random.shuffle(train)
random.shuffle(valid)
random.shuffle(test)

fw1 = open('train_src.txt', 'w')
fw2 = open('train_tgt.txt', 'w')
for t in train:
  fw1.write('{}\n'.format(t.split('\t')[0]))
  fw2.write('{}\n'.format(t.split('\t')[1]))
fw1.close()
fw2.close()

fw1 = open('dev_src.txt', 'w')
fw2 = open('dev_tgt.txt', 'w')
for t in valid:
  fw1.write('{}\n'.format(t.split('\t')[0]))
  fw2.write('{}\n'.format(t.split('\t')[1]))
fw1.close()
fw2.close()

fw1 = open('test_src.txt', 'w')
fw2 = open('test_tgt.txt', 'w')
for t in test:
  fw1.write('{}\n'.format(t.split('\t')[0]))
  fw2.write('{}\n'.format(t.split('\t')[1]))
fw1.close()
fw2.close()

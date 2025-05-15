import os, json

fs = os.listdir('tourism_conversation/annotations')
for f in fs:
  annos = json.load(open('tourism_conversation/annotations/{}'.format(f)))
  fw = open('extracted/{}'.format(f.split('.')[0]+'.txt'), 'w')
  for anno in annos:
    uttr = anno['utterance']
    query = set()
    for ann in anno['annotation']:
      if ann['query'] != None:
        for key in ann['query']:
          query.add('{}:{}'.format(key, ann['query'][key]))
    query = list(query)
    if query == []:
      query = "null"
    else:
      query = ' '.join(query)
    fw.write('{}\t{}\n'.format(uttr, query))
  fw.close()



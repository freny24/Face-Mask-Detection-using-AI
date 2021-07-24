jsonfiles= []
for i in os.listdir(directory):
    jsonfiles.append(getJSON(os.path.join(directory,i)))
jsonfiles[0]
{'FileName': r'C:\Users\Aryan\PycharmProjects\FACE MASK\face-mask-dataset\Dataset\test\with_mask\5-with-mask.jpg',
 'NumOfAnno': 4,
 'Annotations': [{'isProtected': False,
   'ID': 193452793312540288,
   'BoundingBox': [29, 69, 285, 343],
   'classname': 'face_other_covering',
   'Confidence': 1,
   'Attributes': {}},
  {'isProtected': False,
   'ID': 545570408121800384,
   'BoundingBox': [303, 99, 497, 341],
   'classname': 'face_other_covering',
   'Confidence': 1,
   'Attributes': {}},
  {'isProtected': False,
   'ID': 339053397051370048,
   'BoundingBox': [8, 71, 287, 373],
   'classname': 'hijab_niqab',
   'Confidence': 1,
   'Attributes': {}},
  {'isProtected': False,
   'ID': 100482004994698944,
   'BoundingBox': [296, 99, 525, 371],
   'classname': 'hijab_niqab',
   'Confidence': 1,
   'Attributes': {}}]}
df = pd.read_csv(r"C:\Users\Aryan\PycharmProjects\FACE MASK\train.py")
df.head()
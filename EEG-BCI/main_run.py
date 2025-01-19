import os
from processor import processor
import argparse
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser( description='EEG-BCI')

parser.add_argument('--data_root', default='/Users/yangkaisen/MyProject/EEG-BCI/data', type=str)
parser.add_argument('--save_dir', default='./results')

parser.add_argument('--train_model', default='CNN', help='[CNN, GAN, ...]')
parser.add_argument('--load_model', default='best', type=str, help="load pretrained model for test or training")

parser.add_argument('--subject', default=[1,2,3,4,5,6,7,8,9], type=list)
#parser.add_argument('--subject', default=[1], type=list)
parser.add_argument('--electrodes', default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,10,20,21], type=list)
#parser.add_argument('--electrodes', default=[7,10,11], type=list)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--pred_length', default=16, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--preds', default=32, type=int)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--label_dict', default={'left':0, 'right':1, 'foot':2, 'tongue': 3}, type=dict)

parser.add_argument('--c_feats', default=32, type=int)
parser.add_argument('--out_feats', default=4, type=int)
parser.add_argument('--pool_wight', default=4, type=int)
parser.add_argument('--dropout', default=0.5, type=float)
parser.add_argument('--test_rate', default=0.2, type=int)


args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


print('Loading data #################################')
pro = processor(args)

print('Train start #################################')
pro.train()
print('Test start #################################')
pro.test(training=False)
print('Pred start #################################')
pro.pred()


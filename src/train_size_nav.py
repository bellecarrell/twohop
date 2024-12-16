# Import read_json from wikidata.util
from wikidata.utils import load_json, write_json
# Import defaultdict from collections
from collections import defaultdict
import random

# set DATA_DIR to the path of the data directory: /n/holyscratch01/kempner_lab/Everyone/data/twohop-1/nav
DATA_DIR = '/n/holyscratch01/kempner_lab/Everyone/data/twohop-1/nav'

# read 'task.json' in DATA_DIR
task = load_json(DATA_DIR + '/train.json')

# print size of task
print(len(task))
import random
from faker import Faker
from babel.dates import format_date
from tqdm import tqdm
    
def get_datetime_dataset(m):

    fake = Faker()

    FORMATS = [
        # standard named formats
        'short',
        'medium',
        'long',
        'full',

        # custom patterns – year/month/day in varying orders & styles
        'yyyy-MM-dd',
        'yy-MM-dd',
        'MM/dd/yyyy',
        'M/d/yy',
        'dd.MM.yyyy',
        'd.M.yy',
        'MMMM d, yyyy',
        'MMM d yyyy',
        'd MMMM yyyy',
        'd MMM yyyy',
        'EEEE, MMMM d, yyyy',
        'EEE, MMM d, yy',
        'yyyy/MM/dd',
        'yyyy.MM.dd G',
        'd-MMM-yyyy',
        'd MMM, yyyy',
        'MMM d, yy',
        'MMMM dd, yyyy',
        'dd MMMM yyyy',
        'dd, MMM yyyy',
        'yyyy-MM-d',
        'd/MM/yyyy',
        'MM-dd-yy',
        'yy.MM.d',
    ]


    dataset = []

    for _ in tqdm(range(m)):
        dt = fake.date_object()
        human = format_date(dt, format=random.choice(FORMATS), locale='en_US')
        human = human.lower()
        machine = dt.isoformat()
        dataset.append((human, machine))

    return dataset


import json

data = get_datetime_dataset(1000)   # list of (human, machine)

# Convert tuples → dicts for clean JSON
data_json = [{"human": h, "machine": m} for h, m in data]

with open("test.json", "w") as f:
    json.dump(data_json, f, indent=2)
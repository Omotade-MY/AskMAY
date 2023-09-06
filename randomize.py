import random
import numpy as np

def randomName():
    alphanum_code = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                  'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                  'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                  'Y', 'Z']
    alphanum_code.extend([l.lower() for l in alphanum_code])
    alphanum_code.extend(list(range(0,10)))

    return ''.join([str(a) for a in random.choices(alphanum_code, k=10)])

#print('Choices: ',randomName())

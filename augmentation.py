from scipy import ndimage
import random
def random_rotate(x):
    angle=random.randint(-20,20)
    axes= random.sample((0,1,2),k=2)
    rotated_x=ndimage.rotate(x, angle=angle, axes=axes, reshape=False, order=3, mode='constant')
    return rotated_x

def random_shift(x):
    shift=[random.randint(-10,10),random.randint(-10,10),random.randint(-10,10)]
    shifted_x=ndimage.shift(x, shift=shift, order=3, mode='constant', cval=0.0)
    return shifted_x

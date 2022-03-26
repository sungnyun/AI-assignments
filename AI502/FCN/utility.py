from PIL import Image
import numpy as np
import torch


class Rescale(object):
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, img):
        width, height = img.size
        width = self.scale * (int(width/self.scale) - (int(width/self.scale) % 2))
        height = self.scale * (int(height/self.scale) - (int(height/self.scale) % 2))
        img = img.resize((width, height))
        return img
    
'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
    

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()
    
    
def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def mask_output(output):
    _, idx = torch.max(output, dim=1)
    predict = colorize_mask(idx[0].data.cpu().numpy())
    return predict

def evaluate(data, target, n_class):
    # N(i,j): number of pixels of class i predicted to belong to class j
    N = np.zeros((n_class, n_class), dtype=int)
    data = data.view(data.size(0), data.size(1), -1)
    target = target.view(target.size(0), -1)
    _, idx = torch.max(data, dim=1)
    for i in range(idx.size(1)):
        row = target[0][i].item()
        if row == 255:
            row = 0
        col = idx[0][i].item()
        N[row][col] += 1
        
    t = np.zeros(n_class, dtype=int)
    n_ii = np.zeros(n_class, dtype=int)
    mean_acc = 0
    mean_IU = 0
    fw_IU = 0
    n_cl = 0
    for i in range(n_class):
        t[i] = np.sum(N[i])
        if t[i] != 0:
            n_cl += 1
        n_ii[i] = N[i][i]
        
        mean_acc += n_ii[i] / t[i] if t[i] > 0 else 0.
        sum_ji = 0
        for j in range(n_class):
            sum_ji += N[j][i]
        mean_IU += n_ii[i] / (t[i]+sum_ji-n_ii[i]) if (t[i]+sum_ji-n_ii[i]) > 0 else 0.
        fw_IU += t[i] * n_ii[i] / (t[i]+sum_ji-n_ii[i]) if (t[i]+sum_ji-n_ii[i]) > 0 else 0.
        
    mean_acc /= n_cl
    mean_IU /= n_cl
    fw_IU /= np.sum(t)
    pixel_acc = np.sum(n_ii) / np.sum(t)
    
    return pixel_acc, mean_acc, mean_IU, fw_IU
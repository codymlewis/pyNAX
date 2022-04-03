import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation

import numpy as np
from tqdm import tqdm

canvas = np.zeros((2048, 2048, 4))
dims = canvas.shape[:2]
rng = np.random.default_rng(1)
fig, ax = plt.subplots()
ax.axis('off')

class Item:
    def __init__(self, image_fn, h_cen, w_cen, colour=[0, 0, 0]):
        self.colour = np.array(colour, dtype=np.float64)
        self._data = mpimg.imread(image_fn)
        self.h_center = h_cen
        self.w_center = w_cen

    @property
    def data(self):
        data = self._data.copy()
        image = data[:, :, 3]
        for i, c in enumerate(self.colour):
            data[:, :, i] = image * c
        return data

    @property
    def h(self):
        return slice(self.h_center - self.data.shape[0] // 2, self.h_center + self.data.shape[0] // 2)
    
    @property
    def w(self):
        return slice(self.w_center - self.data.shape[1] // 2, self.w_center + self.data.shape[1] // 2)
    
    @property
    def centers(self):
        return np.array([self.h_center, self.w_center])

def add_item(canvas, item):
    if canvas[item.h, item.w, :].any():
        cmask = canvas[item.h, item.w, :] > 0
        data = item.data
        dmask = data > 0
        data[cmask & dmask] = (canvas[item.h, item.w, :][cmask & dmask] + data[cmask & dmask]) / 2
        data[cmask & ~dmask] = canvas[item.h, item.w, :][cmask & ~dmask]
        canvas[item.h, item.w, :] = data
    else:
        canvas[item.h, item.w, :] = item.data
    return canvas

def plot(canvas):
    plt.axis('off')
    plt.imshow(canvas)
    plt.show()

server = Item('images/server.png', 256, dims[1] // 2)
global_model = Item('images/brain-half.png', server.h_center + 400, server.w_center, colour=[1, 1, 1])
clients = [Item('images/mobile.png', dims[0] - 256, (i + 1) * dims[1] // 4) for i in range(3)]
data = [Item('images/database-half.png', clients[i].h_center - 400, clients[i].w_center, colour=rng.uniform(0, 1, (3,))) for i in range(3)]
models = []

items = [server, global_model] + clients + data
for item in items:
    canvas = add_item(canvas, item)
image = ax.imshow(canvas)
target = np.array((global_model.h_center + 300, global_model.w_center))
phase = 0

def update(frame_number):
    global phase
    global target_colour
    if phase == 0:
        for d in data:
            canvas[d.h, d.w, :] = 0
            d.h_center += 0 if (p := (target[0] - d.h_center)) == 0 else int(p / abs(p))
            d.w_center += 0 if (p := (target[1] - d.w_center)) == 0 else int(p / abs(p))
        if np.array(np.linalg.norm(d.centers - target, ord=2) <= 0.001).any():
            avg_colour = np.array([d.colour for d in data]).mean(axis=0)
            [items.pop() for i in range(len(data))]
            data.clear()
            data.append(Item('images/database-half.png', target[0], target[1], colour=avg_colour)) 
            items.append(data[0])
            phase = 1
    elif phase == 1:
        d = data[0]
        canvas[d.h, d.w, :] = 0
        d.h_center += 0 if (p := (global_model.h_center - d.h_center)) == 0 else int(p / abs(p))
        d.w_center += 0 if (p := (global_model.w_center - d.w_center)) == 0 else int(p / abs(p))
        if np.array(np.linalg.norm(d.centers - global_model.centers, ord=2) <= 0.001).any():
            target_colour = d.colour.copy()
            phase = 2
    elif phase == 2:
        d = data[0]
        global_model.colour += np.array([0.01 * (p / abs(p)) if abs((p := tc - gc)) > 0.01 else 0 for gc, tc in zip(global_model.colour, target_colour)])
        d.colour[:] = np.clip(d.colour + 0.01, 0, 1)
        if (abs(target_colour - global_model.colour) < 0.01).all():
            data.clear()
            items.pop()
            models.extend([Item('images/brain-half.png', global_model.h_center, global_model.w_center, colour=global_model.colour) for _ in clients])
            items.extend(models)
            phase = 3
    elif phase == 3:
        for i, m in enumerate(models):
            canvas[m.h, m.w, :] = 0
            m.h_center += 0 if (p := ((clients[i].h_center - 400) - m.h_center)) == 0 else int(p / abs(p))
            m.w_center += 0 if (p := (clients[i].w_center - m.w_center)) == 0 else int(p / abs(p))
            if np.array([np.linalg.norm(m.centers - c.centers, ord=2) <= 0.001 for m, c in zip(models, clients)]).any():
                phase = 4
    for item in items:
        add_item(canvas, item)
    image.set_data(canvas)

frames = 1800
animation = FuncAnimation(fig, update, frames=frames, interval=1, blit=True)
print("Saving animation...")
pbar = tqdm(range(frames))
animation.save('test.mp4', fps=320, progress_callback=lambda i, n: pbar.update(), extra_args=['-vcodec', 'libx264'])
pbar.close()
print("Done!")
import numpy as np




def extract_features(gen, n_img, model):
    no_rounds = int(n_img/gen.batch_size)
    assert(no_rounds == n_img/gen.batch_size, "No of images not divisible by Batch Size")
    out_shape = model.output_shape
    _,x1,x2,x3 = out_shape
    x_shape = (n_img, x1,x2,x3)
    
    y = np.zeros(n_img)
    x = np.zeros(x_shape)
    
    for i in range(no_rounds):
        x_b, y_b = gen.next()
        pred = model.predict(x_b)

        x[(i)*batch_size: (i+1)*batch_size] = pred
        y[(i)*batch_size: (i+1)*batch_size] = y_b
        
    return x,y

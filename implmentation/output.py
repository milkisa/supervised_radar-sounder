import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

def save_output(s_inputs,pred,labels, num,fold,case):

# print(s_inputs.shape)
    
    predict = pred
    predict = F.softmax(predict, dim=1)

    # Get predicted class per pixel
    predict = torch.argmax(predict, dim=1)  # shape: [B, H, W]
  
    # Remove batch dimension if B=1
    predict = predict.squeeze()
    label = labels.squeeze()

    # Convert to numpy for further processing
    
    predict_np = predict.cpu().numpy()
    if case=='test':
     
        s_inputs = s_inputs.cpu().numpy()
        #predict_np= np.argmax(predict, axis=0)
    
        rgb= color_mapping(predict_np)
        #print(rgb.shape)
        rgb= rgb.astype(np.uint8)

        label_rgb= color_mapping(label)

        label_rgb= label_rgb.astype(np.uint8)
        """
        p= predict_np
    

        
        p = torch.from_numpy(p).float()     # convert to torch tensor

        p = p.unsqueeze(0).unsqueeze(0)   # now [1, 1, 360, 250]
        p= F.interpolate(p, size=(1000, 250), mode='nearest')  # resize to original size [1, 1, 1000, 250]
        p = p.squeeze(0).squeeze(0).numpy() # shape [1000, 250]
        """

    
        name= "/home/milkisayebasse/supervised/result/merged_fullsize/aspp/"+str(fold)+ "/" + str(num) + '.png'
        # names= str(num) + "rtoated.png"
        label_names= "/home/milkisayebasse/supervised/result/merged_fullsize/aspp/" +str(fold)+ "/" +str(num) + "_labels.png"
    # dense_names= "D:/important/phd/project/scribble/efficent_u2net/result/scribble/26_aug/" +str(num) + "_dense.png"
        # # # plt.imsave(name,rgb)
        pred_names= "/home/milkisayebasse/supervised/result/merged_fullsize/aspp/" +str(fold)+ "/" +str(num) + "_pred.png"
        plt.imsave(pred_names, rgb)
        plt.imsave(label_names, label_rgb)
        plt.imsave(name,s_inputs.squeeze())


    return predict_np, label
def color_mapping(predict_np):
    green = [0, 255, 0]    # Green
    yellow = [255, 255, 0] # Yellow
    red = [255, 0, 0]      # Red
    blue = [0, 0, 255]     # Blue
    purple = [75,0, 130] # Purple
    orange = [255, 165, 0] # Orange
    # # print(predict_np.shape[0])
    rgb= np.zeros((predict_np.shape[0], predict_np.shape[1], 3), dtype=int)
    rgb[predict_np==6,:]= purple
    rgb[predict_np==3,:]= blue
    rgb[predict_np==2,:]= green
    rgb[predict_np==1,:]= yellow
    rgb[predict_np==4,:]= orange
    rgb[predict_np==5,:]= red
    return rgb;  




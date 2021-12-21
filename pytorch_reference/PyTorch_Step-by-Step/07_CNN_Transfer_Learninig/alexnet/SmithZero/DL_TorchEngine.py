""" (ref) https://github.com/dvgodoy/PyTorchStepByStep/blob/master/Chapter02.1.ipynb
"""
import random 
import logging 

import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import wandb 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch_lr_finder import LRFinder



class D2torchEngine(object): 
    def __init__(self, model, loss_fn, optimizer):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model = model
        self.model.to(self.device)

        self.loss_fn = loss_fn 
        self.optimizer = optimizer 

        # ===
        self.train_loader = None 
        self.val_loader = None 
        self.scheduler = None  # learning_rate scheduler 
        self.is_batch_lr_scheduler = False  # scheduler ON/OFF 
        self.clipping = None # gradient clipping 
        self.wandb = None # W&B board 

        # === visualization {filters, maps, hooks}
        self.visualization = {}
        self.handles = {}


        # === 
        self.train_losses = [] 
        self.val_losses = [] 
        self.learning_rates = [] 
        self.total_epochs = 0 

        # === 
        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()


    def to(self, device):
        # === This method allows the user to specify a different device === # 
        
        self.device = device 
        self.model.to(self.device)    

    def set_loaders(self, train_loader, val_loader):
        # === This method allows the user to define which train_loader (and val_loader) to use ===#
        
        self.train_loader = train_loader 
        self.val_loader = val_loader

    def set_wandb(self, wandb): 
        # === This method allows the user to use a W&B instance === # 
        self.wandb = wandb


    def _make_train_step(self): 
        # === build this in higher-order function === # 
        def perform_train_step(input, label):
            self.model.train()  # set train mode 

            yhat = self.model(input) # get score out of the model 
            loss = self.loss_fn(yhat, label) # computes the loss 

            loss.backward() # computes gradients 

            if callable(self.clipping): # make sure gradient clipping 
                self.clipping()         # after computing gradients 

            # Updates parameters using gradients and the learning rate 
            self.optimizer.step() 
            self.optimizer.zero_grad() 
            
            # Returns the loss 
            return loss.item() 
        return perform_train_step

    def _make_val_step(self): 
        # === build this in higher-order function === # 
        def perform_val_step(input, label): 
            self.model.eval() # set eval mode 

            yhat = self.model(input)
            loss = self.loss_fn(yhat, label)

            return loss.item()
        return perform_val_step

    def _mini_batch(self, validation=False): 
        if validation : 
            data_loader = self.val_loader
            step = self.val_step
        else: 
            data_loader = self.train_loader 
            step = self.train_step 
        
        if data_loader is None: 
            print(f"No any dataloader @ validation={validation}")
            return None 
            
        n_batches = len(data_loader)
        
        # === Run loop === # 
        mini_batch_losses = [] 
        for i, (x_batch, y_batch) in enumerate(data_loader): 
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step(x_batch, y_batch) # train/val-step
            mini_batch_losses.append(mini_batch_loss)

            if not validation: # only during training! 
                self._mini_batch_schedulers(i/n_batches)# call the learning rate scheduler 
                                                        # at the end of every mini-batch update 
        # Return the avg. loss 
        return np.mean(mini_batch_losses) 

    def set_seed(self, seed:int=42): 
        # === Reproducible === #
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False    
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        try:
            # === sampling for imbalanced data === # 
            self.train_loader.sampler.generator.manual_seed(seed) # *** (add later)
        except AttributeError:
            pass 

    def train(self, n_epochs, seed=42):
        self.set_seed(seed) # To ensure reproducibility of the training process

        if self.wandb:
            # Tell wandb to watch what the model gets up to: gradients, weights, and more!
            self.wandb.watch(self.model, self.loss_fn, log="all", log_freq=10)
            self.wandb.define_metric("train_loss", summary="min") # (ref) https://docs.wandb.ai/guides/track/log
            self.wandb.define_metric("val_loss", summary="min")

        for epoch in tqdm(range(n_epochs)):
            self.total_epochs += 1

            # *** TRAINING *** # 
            train_loss = self._mini_batch(validation=False) 
            self.train_losses.append(train_loss)

            # *** VALIDATION *** # 
            # Set no gradients ! 
            with torch.no_grad(): 
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)

            self._epoch_schedulers(val_loss)    # learning_rate scheduler
                                                # make sure to set after validation 

            # *** If a W&B has been set *** # 
            if self.wandb: 
                # Record logs of both losses for each epoch 
                log_dict = {"epoch":epoch, "train_loss":train_loss}
                if val_loss is not None:
                    update_dict = {'val_loss': val_loss} 
                    log_dict.update(update_dict)  # dict() update 
                if self.scheduler is not None: 
                    log_dict.update({'lr_schedule': np.array(self.learning_rates[-1][-1])}) # get the last LR
                self.wandb.log(log_dict)

            # *** Save weights *** # 


    # ====================== # 
    #       Save & Load      # 
    # ====================== # 
    def save_checkpoint(self, filename:str): 
        # Builds dictionary with all elements for resuming training
        checkpoint = {  'epoch': self.total_epochs, 
                        'model_state_dict': self.model.state_dict(), 
                        'optimizer_state_dict': self.optimizer.state_dict(), 
                        'train_loss' : self.train_losses, 
                        'val_loss' : self.val_losses
                    }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename:str): 
        checkpoint = torch.load(filename) # Loads dictionary 

        # === Restore state for model and optimizer === # 
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.total_epochs = checkpoint['epoch']
        self.train_losses = checkpoint['train_loss']
        self.val_losses = checkpoint['val_loss']

        # always use TRAIN for resuming training  
        self.model.train()

    # ====================== # 
    #     1:1 Prediction     # 
    # ====================== # 
    def predict(self, input):
        self.model.eval() # Set is to evaluation mode for predictions
        
        input_tensor = torch.as_tensor(input).float() # Takes a Numpy input and make it a float tensor 
        yhat_tensor = self.model(input_tensor.to(self.device)) # Send input to device and uses model for prediction
        
        self.model.train() # Set it back to train mode

        return yhat_tensor.detach().cpu().numpy()

    # ====================== # 
    #        Plotting        #  
    # ====================== # 
    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label='Training Loss', c='b')
        plt.plot(self.val_losses, label='Validation Loss', c='r')
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend() 
        plt.tight_layout()
        return fig 

    # ====================== # 
    #         Metrics        # 
    # ====================== # 
    def _metrics(self, name:str):
        """ name can be: 
                - 'accuracy'
            Usage example: 
                - metric_dict['accuracy'] 
        """
        def top1_accuracy(predictions, labels, threshold=0.5):
            n_samples, n_dims = predictions.size() # (num_instances, num_classes)

            if n_dims > 1: 
                # *** multiclass classficiation *** # 
                _, argmax_idx = torch.max(predictions, dim=1) 
            else: 
                # *** binary classficiation *** # 
                # we NEED to check if the last layer is a sigmoid (to produce probability)
                if isinstance(self.model, nn.Sequential) and isinstance(self.model[-1], nn.Sigmoid):
                    argmax_idx = (predictions > threshold).long()
                else: 
                    argmax_idx = (torch.sigmoid(predictions) > threshold).long()

            # How many samples got classified correctly for each class 
            result = [] 
            for cls_idx in range(n_dims):
                n_class = (labels == cls_idx).sum().item() # nun_items for each class 
                n_correct = (argmax_idx[labels == cls_idx ] == cls_idx).sum().item() # num_corrects for each class 

#                print(labels == cls_idx) # where is the label ? 
#                print(argmax_idx[labels == cls_idx]) # what is the prediction results on each label 
#                print(argmax_idx[labels == cls_idx ] == cls_idx) # return only the correct answers 

                result.append((n_correct, n_class))
            return torch.tensor(result)
            
        # *** Metric Switch *** # 
        metric_dict = dict( accuracy=top1_accuracy,
                            )
        return metric_dict[name]

    def correct(self, input, labels):
        self.model.eval() 
        yhat = self.model(input.to(self.device))
        labels = labels.to(self.device)

        # === 
        metric_func = self._metrics('accuracy')

        results = metric_func(yhat, labels)

        return results

    @staticmethod 
    def loader_apply(dataloader, func, reduce='sum'):
        results = [func(inputs, labels) for idx, (inputs, labels) in enumerate(dataloader)]
        results = torch.stack(results, axis=0)

        if reduce == 'sum': 
            results = results.sum(axis=0)
        elif reduce == 'mean': 
            results = results.float().mean(axis=0)
        return results     

    # ====================== # 
    #   Learninig Scheduler  #  
    # ====================== # 
    def set_optimizer(self, optimizer): 
        self.optimizer = optimizer

    def set_lr_scheduler(self, scheduler): 
        
        if scheduler.optimizer == self.optimizer:
            self.scheduler = scheduler
            if (isinstance(scheduler, optim.lr_scheduler.CyclicLR) or 
                isinstance(scheduler, optim.lr_scheduler.OneCycleLR) or 
                isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)):
                self.is_batch_lr_scheduler = True 
            else:
                self.is_batch_lr_scheduler = False


    def _epoch_schedulers(self, val_loss):
        if self.scheduler: 
            if not self.is_batch_lr_scheduler: # if not batch_scheduler 
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step() 

                current_lr = list(map(lambda d: d['lr'], self.scheduler.optimizer.state_dict()['param_groups'])) 
                self.learning_rates.append(current_lr) # log of learninig_rates 

    def _mini_batch_schedulers(self, frac_epoch): 
        if self.scheduler: 
            if self.is_batch_lr_scheduler: # if batch_scheduler 
                if isinstance(self.scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts): 
                    self.scheduler.step(self.total_epochs + frac_epoch)
                else: 
                    self.scheduler.step()

                current_lr = list(map(lambda d:d['lr'], self.scheduler.optimizer.state_dict()['param_groups']))
                self.learning_rates.append(current_lr) # log of learninig_rates 

    def lr_range_test(self, end_lr=1e-1, num_iter=100): 
        # === Learning Rate Range Test === # 
        # Using LRFinder 
        assert self.train_loader is not None, "You didn't set trainloader"
        
        fig, ax = plt.subplots(1, 1, figsize=(6,4))

        lr_finder = LRFinder(self.model, self.optimizer, self.loss_fn, device=self.device)
        lr_finder.range_test(self.train_loader, end_lr=end_lr, num_iter=num_iter)
        lr_finder.plot(ax=ax, log_lr=True)

        fig.tight_layout()
        lr_finder.reset()

        return fig 

    # ======================== #
    #     Gradient Clipping    #        
    # ======================== #
    def set_clip_grad_value(self, clip_value=1.0): 
        # *** gradient value clipping *** # 
        self.clipping = lambda: nn.utils.clip_grad_value_(self.model.parameters(), clip_value=clip_value)


    def set_clip_grad_norm(self, max_norm=1.0, norm_type=2):
        # *** gradient norm clipping *** # 
        self.clipping = lambda: nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_norm, norm_type=norm_type)


    def set_clip_backprop(self, clip_value=1.0): 
        # *** clipping during backpropagation *** # 

        if self.clipping is None: 
            self.clipping = []

        for parm in self.model.parameters():
            if parm.requires_grad:
                func = lambda grad: torch.clamp(grad, -clip_value, clip_value)
                handle = parm.register_hook(func)
                self.clipping.append(handle)

    def remove_clip(self):
        # === Reset === # 
        # *** for the clip_backprop *** # 
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        # *** for the clip_grad_value/norm *** #             
        self.clipping = None  


    # ======================== #
    #     Visualize Tensors    #        
    # ======================== #
    @staticmethod
    def _visualize_tensors(axs, x, y=None, yhat=None, layer_name='', title=None):
        # The number of images is the number of subplots in a row
        n_images = len(axs)
        # Gets max and min values for scaling the grayscale
        minv, maxv = np.min(x[:n_images]), np.max(x[:n_images])
        # For each image
        for j, image in enumerate(x[:n_images]):
            ax = axs[j]
            # Sets title, labels, and removes ticks
            if title is not None:
                ax.set_title('{} #{}'.format(title, j), fontsize=12)
            ax.set_ylabel(
                '{}\n{}x{}'.format(layer_name, *np.atleast_2d(image).shape), 
                rotation=0, labelpad=40
            )
            xlabel1 = '' if y is None else f'\nLabel: {y[j]}'
            xlabel2 = '' if yhat is None else f'\nPredicted: {yhat[j]}'
            xlabel = f'{xlabel1}{xlabel2}'
            if len(xlabel):
                ax.set_xlabel(xlabel, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

            # Plots weight as an image
            ax.imshow(
                np.atleast_2d(image.squeeze()),
                cmap='viridis', 
                vmin=minv, 
                vmax=maxv
            )
        return

    # ======================== #
    #     Visualize Filters    #        
    # ======================== #        
    def visualize_filters(self, layer_name, **kwargs):
        try:
            # Gets the layer object from the model
            layer = self.model
            for name in layer_name.split('.'):
                layer = getattr(layer, name)
            # We are only looking at filters for 2D convolutions
            if isinstance(layer, nn.Conv2d):
                # Takes the weight information
                weights = layer.weight.data.cpu().numpy()
                # weights -> (channels_out (filter), channels_in, H, W)
                n_filters, n_channels, _, _ = weights.shape

                # Builds a figure
                size = (2 * n_channels + 2, 2 * n_filters)
                fig, axes = plt.subplots(n_filters, n_channels, figsize=size)
                axes = np.atleast_2d(axes)
                axes = axes.reshape(n_filters, n_channels)
                # For each channel_out (filter)
                for i in range(n_filters):    
                    D2torchEngine._visualize_tensors(
                        axes[i, :],
                        weights[i],
                        layer_name=f'Filter #{i}', 
                        title='Channel'
                    )

                for ax in axes.flat:
                    ax.label_outer()

                fig.tight_layout()
                return fig
        except AttributeError:
            return


    # ======================== #
    #          Hooks           #        
    # ======================== #        
    def attach_hooks(self, layers_to_hook, hook_fn=None):

        self.visualization = {} # Clear any previous values

        # *** Creates the dictionary to map layer objects to their names *** # 
        modules = list(self.model.named_modules()) 
        layer_names = {layer: name for name, layer in modules[1:]}

        if hook_fn is None:
            # *** Hook function to be attached to the forward pass *** # 
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer] # Gets the layer name
                values = outputs.detach().cpu().numpy() # for visualization

                if self.visualization[name] is None:
                    self.visualization[name] = values                
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            if name in layers_to_hook: # If the layer is in our list
                self.visualization[name] = None # Initializes the corresponding key in the dictionary
                self.handles[name] = layer.register_forward_hook(hook_fn) # Register the forward hook and keep the handle in another dict

    def remove_hooks(self):
        # Loops through all hooks and removes them
        for handle in self.handles.values():
            handle.remove()
        # Clear the dict, as all hooks have been removed
        self.handles = {}


    def visualize_outputs(self, layers, n_images=10, y=None, yhat=None):
        layers = filter(lambda l: l in self.visualization.keys(), layers)
        layers = list(layers)
        shapes = [self.visualization[layer].shape for layer in layers]
        n_rows = [shape[1] if len(shape) == 4 else 1 for shape in shapes]
        total_rows = np.sum(n_rows)

        fig, axes = plt.subplots(total_rows, n_images, 
                                figsize=(1.5*n_images, 1.5*total_rows))
        axes = np.atleast_2d(axes).reshape(total_rows, n_images)

        # Loops through the layers, one layer per row of subplots
        row = 0
        for i, layer in enumerate(layers):
            start_row = row
            # Takes the produced feature maps for that layer
            output = self.visualization[layer]

            is_vector = len(output.shape) == 2

            for j in range(n_rows[i]):
                D2torchEngine._visualize_tensors(
                    axes[row, :],
                    output if is_vector else output[:, j].squeeze(),
                    y, 
                    yhat, 
                    layer_name=layers[i] \
                               if is_vector \
                               else f'{layers[i]}\nfil#{row-start_row}',
                    title='Image' if (row == 0) else None
                )
                row += 1

        for ax in axes.flat:
            ax.label_outer()

        plt.tight_layout()
        return fig
        



        

                






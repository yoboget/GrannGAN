import torch
from eval.evaluation import Eval
from eval.mmd_rbf import MMD_RBF
from utils.func import get_symetric_sparse_adj, discretize, normalize_adj, JSD_distance
import time

class Trainer():    
    def __init__(self, trainloader, 
                 testloader,
                 models, 
                 optimizors,               
                 schedulers,
                 args
                 ):
        
        self.trainloader = trainloader  
        self.testloader = testloader
        self.generator, self.discriminator = models
        self.opt_generator, self.opt_discriminator = optimizors
        self.scheduler_g, self.scheduler_d = schedulers
        self.epochs = args.epochs
        self.args = args
        self.sizes = args.sizes
        self.device = args.device
        
        if not self.args.debug:
            self.eval = Eval(args, self.generator, self.discriminator)
            self.eval.init_history_batch()
        self.mmd_rbf = MMD_RBF(self.sizes['node_types']+self.sizes['cycles'], 
                               self.sizes['edge_types']).to(args.device)
        
                  
    def fit(self):
        print('Training starts...')
        n_iter = 0
        for e in range(self.epochs):         
            print(f'Epoch {n_iter+1} starts...')
            start = time.time()
            for (i, batch) in enumerate(self.trainloader):               
                batch = batch.to(self.device)
                if self.args.train_step == 'adja':            
                    self.discriminator_step_adja(batch)
                    self.generator_step_adja(batch)                    
                        
                elif self.args.train_step == 'nodes':            
                    self.discriminator_step_node(batch)         
                    self.generator_step_node(batch)
                        
                elif self.args.train_step == 'scaffold':            
                    self.discriminator_step_scaffold(batch)
                    self.generator_step_scaffold(batch)                       
                else: raise(Exception('Training type not implemented'))
            
                end = time.time()
                
                n_iter +=1
                if n_iter % 1000 == 0: 
                    end = time.time()
                    print('1000 iter time: ', end-start)
                    self.scheduler_g.step()
                    self.scheduler_d.step()
                    self.args.tau = self.args.tau*self.args.tau_decay
          
                    if not self.args.debug:                                 
                        self.eval.add_epoch_loss(n_iter)
                        batch = next(iter(self.testloader)).to(self.device)
                        if self.args.dataset == 'Fingerprint':
                            if self.args.train_step == 'adja': 
                                self.eval.step_edge_attr(self.generator,                                      
                                                      batch, 
                                                      self.sizes, 
                                                      self.args, 
                                                      n_iter)
                            elif self.args.train_step == 'nodes':    
                                self.eval.step_nodes_attr(self.generator,
                                                      batch, 
                                                      self.sizes,
                                                      self.args, 
                                                      n_iter
                                                      )
                        else:
                            if self.args.train_step == 'adja': 
                                self.eval.step_edge_types(self.generator,                                      
                                                      batch, 
                                                      self.sizes, 
                                                      self.args, 
                                                      n_iter)
                            elif self.args.train_step == 'nodes':    
                                self.eval.step_nodes(self.generator,
                                                      batch, 
                                                      self.sizes,
                                                      self.args, 
                                                      n_iter
                                                      )
                       
                    start = time.time()
               

    def discriminator_step_adja(self, batch):
        self.discriminator.train()
        self.generator.eval()      
        self.opt_discriminator.zero_grad()

        
        z = torch.randn(batch.edge_index.shape[-1], 
                        self.sizes['noise']).to(self.device)       
        z = get_symetric_sparse_adj(batch.edge_index, z)
       
        if self.args.normalize:
            norm = normalize_adj(batch.edge_index, batch.x)
        else: norm = None
        
        if self.args.cycles:
            x = torch.cat((batch.x, batch.cycles), dim = -1)
        else: x = batch.x

        edge_attr_gen  = self.generator(batch.edge_index, x, 
                                        z, norm = norm)

        
        if self.args.discretizing_method:                                        
            edge_attr_gen = discretize(edge_attr_gen, 
                                       method = self.args.discretizing_method)
        else:
            edge_attr_gen = torch.tanh(edge_attr_gen)
        
        edge_attr_gen = get_symetric_sparse_adj(batch.edge_index, 
                                                edge_attr_gen, 
                                                method = 'keep_upper_tri')
        

        score_gen = self.discriminator(batch.edge_index, x,
                                       edge_attr_gen, norm = norm)
        score_real = self.discriminator(batch.edge_index, x, 
                                        batch.edge_attr, norm = norm)

        loss = -.5*(score_real.mean() - score_gen.mean()) 
        loss.backward()
        self.opt_discriminator.step()
        
        #sample_real = (x, batch.edge_index,  batch.batch, batch.edge_attr)
        #sample_gen = (x, batch.edge_index,  batch.batch, edge_attr_gen)
        #mmd = self.mmd_rbf(sample_real, sample_gen)
        #print(mmd)

        if not self.args.debug:
            self.eval.history_temp['loss'].append(-loss.item())
 
    def generator_step_adja(self, batch):
        self.discriminator.eval()
        self.generator.train()
        self.opt_generator.zero_grad()
        
        z = torch.randn(batch.edge_index.shape[-1], 
                        self.sizes['noise']).to(self.device)       
        z = get_symetric_sparse_adj(batch.edge_index, z)
       
        if self.args.normalize:
            norm = normalize_adj(batch.edge_index, batch.x)
        else: norm = None
        if self.args.cycles:
            x = torch.cat((batch.x, batch.cycles), dim = -1)
        else: x = batch.x
        
        edge_attr_gen  = self.generator(batch.edge_index, x,
                                        z, norm = norm)
        edge_attr_gen = get_symetric_sparse_adj(batch.edge_index, 
                                                edge_attr_gen, 
                                                method = 'keep_upper_tri')
        
       
        if self.args.discretizing_method:                                        
            edge_attr_gen = discretize(edge_attr_gen, 
                                       method = self.args.discretizing_method)
        else:
            edge_attr_gen = torch.tanh(edge_attr_gen)
                  
        loss = -self.discriminator(batch.edge_index, x, 
                                   edge_attr_gen,  norm = norm).mean()            
        loss.backward()
        self.opt_generator.step()

    def discriminator_step_node(self, batch):
        self.discriminator.train()
        self.generator.eval()      
        self.opt_discriminator.zero_grad()

        z = torch.randn(batch.x.shape[0], 
                        self.sizes['noise']).to(self.device)      
        if self.args.normalize:
            norm = normalize_adj(batch.edge_index, batch.x)
        else: norm = None
        if self.args.cycles:
            z = torch.cat((z, batch.cycles), dim = -1)
            x = torch.cat((batch.x, batch.cycles), dim = -1)
        else: x = batch.x
        
        x_gen  = self.generator(batch.edge_index, z, norm = norm)
        if self.args.discretizing_method:                                        
            x_gen = discretize(x_gen, method = self.args.discretizing_method)
        else: 
            x_gen = torch.tanh(x_gen)
        
        if self.args.cycles:
            x_gen = torch.cat((x_gen, batch.cycles), dim = -1)
        
        score_gen = self.discriminator(batch.edge_index, x_gen, norm = norm).mean()
        score_real = self.discriminator(batch.edge_index, x, norm = norm).mean()
        
        loss = -0.5*(score_real - score_gen) 
        loss.backward()
        self.opt_discriminator.step()
        
        #sample_real = (x, batch.edge_index, batch.batch, None)
        #sample_gen = (x_gen, batch.edge_index, batch.batch, None)
        #mmd = self.mmd_rbf(sample_real, sample_gen)
        #print(mmd)

        if not self.args.debug:
            self.eval.history_temp['loss'].append(-loss.item())
        
    def generator_step_node(self, batch):
        self.discriminator.eval()
        self.generator.train()
        self.opt_generator.zero_grad()
  
        z = torch.randn(batch.x.shape[0], 
                        self.sizes['noise']).to(self.device)      
        if self.args.normalize:
            norm = normalize_adj(batch.edge_index, batch.x)
        else: norm = None
        if self.args.cycles:
            z = torch.cat((z, batch.cycles), dim = -1)

        x_gen  = self.generator(batch.edge_index, z, norm = norm)
        if self.args.discretizing_method:                                        
            x_gen = discretize(x_gen, method = self.args.discretizing_method)
        else:
            x_gen = torch.tanh(x_gen)
        if self.args.cycles:
            x_gen = torch.cat((x_gen, batch.cycles), dim = -1)
        
        loss = -self.discriminator(batch.edge_index, x_gen, norm = norm).mean()
        loss.backward()
        self.opt_generator.step()
        

        
    def discriminator_step_scaffold(self, batch):
        self.discriminator.train()
        self.generator.eval()      
        self.opt_discriminator.zero_grad()
        print('batch starts...')
    
        z = torch.randn(self.sizes['batch'],                
                         self.sizes['mols'],
                         self.sizes['noise']).to(self.device)
        scaffold = batch[1][:,:4].sum(1).unsqueeze(1)
        if self.args.cycles:
            cycles = batch[1][:, 5:]
            scaffold = torch.cat([scaffold, cycles], dim = 1)
        x_gen = self.generator(z)
        scaffold_generated = discretize_scaffold(x_gen, self.args)
        score_gen = self.discriminator(scaffold_generated).mean()

        score_real = self.discriminator(scaffold).mean()

        loss = -(score_real - score_gen) 
        loss.backward()
        self.opt_discriminator.step()
        
        if not self.args.debug:
            self.eval.history_temp['loss'].append(-loss.item())
        
    def generator_step_scaffold(self, batch):
        self.discriminator.eval()
        self.generator.train()
        self.opt_generator.zero_grad()

        
        z = torch.randn(self.sizes['batch'],                
                         self.sizes['mols'],
                         self.sizes['noise']).to(self.device)
        
        logit = self.generator(z)
        scaffold_generated = func.discretize_scaffold(logit, self.args)

        loss = -self.discriminator(scaffold_generated).mean()
        loss.backward()
        self.opt_generator.step()

    def batch_to_device(self, batch):
        if self.args.cycles:
            annotation_batch, adjacency_batch, cycles = batch
            annotation_batch = annotation_batch.to(self.device).float()
            adjacency_batch = adjacency_batch.to(self.device).float()
            cycles_batch = cycles.to(self.device).float()
            return annotation_batch, adjacency_batch, cycles_batch
        else:
            annotation_batch, adjacency_batch = batch    
            annotation_batch = annotation_batch.to(self.device).float()
            adjacency_batch = adjacency_batch.to(self.device).float()
            return annotation_batch, adjacency_batch
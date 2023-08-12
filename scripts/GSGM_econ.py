import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
#import horovod.tensorflow.keras as hvd
import utils_econ
from deepsets_econ import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu

# tf and friends
tf.random.set_seed(1235)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM',npart=30,config=None,factor=1):
        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")

        self.activation = layers.LeakyReLU(alpha=0.01)
        #self.activation = swish
        # self.activation = relu
        self.factor=factor
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = self.config['NUM_FEAT']
        self.num_cluster = self.config['NUM_CLUS']
        self.num_cond = self.config['NUM_COND']
        #self.num_cond = self.config['NUM_COND']+1
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.num_steps = self.config['MAX_STEPS']//self.factor
        self.ema=0.999

        self.timesteps =tf.range(start=0,limit=self.num_steps + 1, dtype=tf.float32) / self.num_steps + 8e-3 
        alphas = self.timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = tf.math.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        self.betas = tf.clip_by_value(betas, clip_value_min =0, clip_value_max=0.999)
        alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(alphas, 0)
        alphas_cumprod_prev = tf.concat((tf.ones(1, dtype=tf.float32), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * tf.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - self.alphas_cumprod)
        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank
        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond)) # shape=(None, 2) 2 print('inputs_cond',inputs_cond)
        inputs_cluster = Input((self.num_cluster)+54) # Replace with self.num_z_layers #  shape=(None, 2) 2 print('inputs_cluster', inputs_cluster)
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects       

        graph_conditional = self.Embedding(inputs_time,self.projection) # shape=(None, 64) print('graph_conditional',graph_conditional)
        cluster_conditional = self.Embedding(inputs_time,self.projection) # shape=(None, 64) print('cluster_conditional',cluster_conditional)      
        
        #print('graph_conditional 0',graph_conditional) # shape=(None, 64)
        #print('cluster_conditional 0',cluster_conditional) # shape=(None, 64)
        
        graph_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [graph_conditional,inputs_cluster,inputs_cond],-1))
        graph_conditional=self.activation(graph_conditional)
        
        cluster_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [cluster_conditional,inputs_cond],-1))
        cluster_conditional=self.activation(cluster_conditional)

        # These outputs for graph_conditional and cluster_conditional are (None,64) as they pass through embedding sizedense layer.
        #print('graph_conditional 1',graph_conditional) # shape=(None, 64)
        #print('cluster_conditional 1',cluster_conditional) # shape=(None, None, 64)
        # These conditionals will now get attached to Inputs
        # x1 + 

        '''
        graph_conditional KerasTensor(type_spec=TensorSpec(shape=(None, 64), dtype=tf.float32, name=None), name='dense_1/BiasAdd:0', description="created by layer 'dense_1'")
        cluster_conditional KerasTensor(type_spec=TensorSpec(shape=(None, 64), dtype=tf.float32, name=None), name='dense_3/BiasAdd:0', description="created by layer 'dense_3'")
        inputs_cluster KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name='input_3'), name='input_3', description="created by layer 'input_3'")
        inputs_cond KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name='input_2'), name='input_2', description="created by layer 'input_2'")
        graph_conditional KerasTensor(type_spec=TensorSpec(shape=(None, 64), dtype=tf.float32, name=None), name='leaky_re_lu/LeakyRelu:0', description="created by layer 'leaky_re_lu'")
        cluster_conditional KerasTensor(type_spec=TensorSpec(shape=(None, 64), dtype=tf.float32, name=None), name='leaky_re_lu/LeakyRelu:0', description="created by layer 'leaky_re_lu'")
        '''

        # This block is only for cells, where the input x is of dim (None,None,4) (x,y,z,E)

        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=1,
            num_transformer = 8,
            projection_dim = 128,
            mask = inputs_mask,
        )

        #print(f"inputs = {inputs}") # inputs = KerasTensor(type_spec=TensorSpec(shape=(None, None, 4), dtype=tf.float32, name='input_5'), name='input_5', description="created by layer 'input_5'")
        #print(f"outputs1 = {outputs}") # outputs1 = KerasTensor(type_spec=TensorSpec(shape=(None, None, 4), dtype=tf.float32, name=None), name='time_distributed_5/Reshape_1:0', description="created by layer 'time_distributed_5'")
        #print('num_cluster', self.num_cluster) # num_cluster 2
        
        # (None, None, 4){x,y,z,E} , (None, 64){time embedding_attaches to each particle differently} , cluster = (None, 2){Number, E_cl} , inputs_cond = (None, 2) {Pgen, Theta}, (None,1){For masking E value cell < 0}
        self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_cluster,inputs_cond,inputs_mask],outputs=outputs)
        # outputs = (None, None, 4)
        # This is used to calculate score (As score dim should be similar to input image dimmension)
        # self.model_part([perturbed_x, random_t,cluster,cond,mask])


        def count_parameters(model):
            return model.count_params()

        num_params = count_parameters(self.model_part) 
        print("Number of parameters: {:,}".format(num_params)) # Number of parameters: 314,372


        # This block is only for cluster, where the input x is of dim (None,2) (Number of cells, cluster_E)

        outputs = Resnet(
            inputs_cluster, # inputs (None, 2)
            self.num_cluster+54, # end_dim , # Replace with self.num_z_layers
            cluster_conditional, # time_embedding
            num_embed=self.num_embed, # num_embed
            num_layer = 5,
            mlp_dim= 512,
        )
        '''
        def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
        ):'''

        print(f"outputs2 = {outputs}") # outputs = KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name=None), name='dense_46/BiasAdd:0', description="created by layer 'dense_46'")
        
        # cluster = (None, 2){ClusterSum, N_Hits} , (None, 64){time embedding_attaches to each particle differently} , inputs_cond = (None, 2) {Pgen, Theta}.
        self.model_cluster = keras.Model(inputs=[inputs_cluster,inputs_time,inputs_cond],outputs=outputs)
        # output = (None, 2)
        # # This is used to calculate score (As score dim should be similar to input image dimmension)
        # self.model_cluster([perturbed_x, random_t,cond])

        num_params = count_parameters(self.model_cluster)
        print("Number of parameters: {:,}".format(num_params)) # Number of parameters: 4,517,762

        print(self.model_part)
        print(self.model_cluster)

        self.ema_cluster = keras.models.clone_model(self.model_cluster)
        self.ema_part = keras.models.clone_model(self.model_part)
        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        #half_dim = 16
        half_dim = self.num_embed // 4
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq


    def Embedding(self,inputs,projection):
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    

    @tf.function
    def train_step(self, inputs):
        
        part,cluster,cond,mask = inputs

        print(part.shape)
        print(cluster.shape)
        print(cond.shape)
        print(mask.shape)

        '''
        <tf.Tensor 'inputs:0' shape=(None, 200, 4) dtype=float32>
        <tf.Tensor 'inputs_1:0' shape=(None, 2) dtype=float32> , <tf.Tensor 'inputs_1:0' shape=(None, 57) dtype=float32>
        <tf.Tensor 'inputs_2:0' shape=(None, 2) dtype=float32>
        <tf.Tensor 'inputs_3:0' shape=(None, 200, 1) dtype=float32>
        '''

        random_t = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32) # None is for the batch size.
        
        alpha = tf.gather(tf.sqrt(self.alphas_cumprod),random_t)
        sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),random_t)
        sigma = tf.clip_by_value(sigma, clip_value_min = 1e-3, clip_value_max=0.999)

        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)

        # cluster solver
        with tf.GradientTape() as tape:
            #cluster
            z = tf.random.normal((tf.shape(cluster)),dtype=tf.float32)
            perturbed_x = alpha*cluster + z * sigma # perturbed_x_2.shape (None, 2) , (None, 57)           
            score = self.model_cluster([perturbed_x,random_t,cond])
            v = alpha * z - sigma * cluster # (None, 57)
            losses = tf.square(score - v)
            loss_cluster = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))

        trainable_variables = self.model_cluster.trainable_variables
        g = tape.gradient(loss_cluster, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))        
            
        for weight, ema_weight in zip(self.model_cluster.weights, self.ema_cluster.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        # particle solver
        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
            perturbed_x = alpha_reshape*part + z * sigma_reshape # perturbed_x.shape (None, 200, 4)
            score = self.model_part([perturbed_x, random_t,cluster,cond,mask])
            
            v = alpha_reshape * z - sigma_reshape * part
            losses = tf.square(score - v)*mask
            
            loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_part.trainable_variables
        g = tape.gradient(loss_part, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))
        
        self.loss_tracker.update_state(loss_cluster + loss_part) # update loss in loss-tracker

        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_cluster":tf.reduce_mean(loss_cluster),
        }

    @tf.function
    def test_step(self, inputs):
        part,cluster,cond,mask = inputs

        random_t = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        #random_t = tf.cast(random_t,tf.float32)
        alpha = tf.gather(tf.sqrt(self.alphas_cumprod),random_t)
        sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),random_t)
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            
        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        perturbed_x = alpha_reshape*part + z * sigma_reshape

        score = self.model_part([perturbed_x, random_t,cluster,cond,mask])
        v = alpha_reshape * z - sigma_reshape * part
        losses = tf.square(score - v)*mask
            
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                    
        #cluster
        z = tf.random.normal((tf.shape(cluster)),dtype=tf.float32)
        perturbed_x = alpha*cluster + z * sigma
        # print(perturbed_x.shape) # most probably (None,2)          
        score = self.model_cluster([perturbed_x, random_t,cond])
        v = alpha * z - sigma * cluster
        losses = tf.square(score - v)
        loss_cluster = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
        self.loss_tracker.update_state(loss_cluster + loss_part)
        
        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_cluster":tf.reduce_mean(loss_cluster),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)

    def generate_cluster(self,cond):
        start = time.time()
        cluster_info = self.DDPMSampler(cond,self.ema_cluster,
                                    data_shape=[self.num_cluster+54], # Replace with self.num_z_layers
                                    const_shape = [-1,1]).numpy()
        end = time.time()
        print("Sampling Clusters in {} Events ({} Seconds)".format(cond.shape[0],end - start))

        return cluster_info


    def generate(self,cond,cluster_info):
        start = time.time()
        print('cond',cond.shape) #cond (500, 2)
        print('self.ema_cluster',self.ema_cluster) # self.ema_cluster <keras.src.engine.functional.Functional object at 0x2aac86fae0a0>
        print(self.num_cluster+54) # 2

        # Are you sure self.ema_cluster has the trained weights for the generation step?
        cluster_info = self.DDPMSampler(cond,self.ema_cluster,
                                    data_shape=[self.num_cluster+54], # # Replace with self.num_z_layers
                                    const_shape = [-1,1]).numpy()
        end = time.time()

        print('cluster_info',cluster_info.shape) # (5,2) , (20,57)
        print('cluster_info',cluster_info[0].shape) # (57,)

        print("Sampling Clusters in {} Events ({} Seconds)".format(cond.shape[0],end - start))

        nparts = np.expand_dims(np.clip(utils_econ.revert_npart(cluster_info[:,-1],self.max_part),
                                        0,self.max_part),-1)
        
        print('nparts',nparts) # nparts [[0] [0] ....]
        print('****************')

        mask = np.expand_dims(
            np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
        assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of cells'
        start = time.time()
        #print(mask.shape) # (20, 200, 1)
        parts = self.DDPMSampler(tf.convert_to_tensor(cond,dtype=tf.float32),
                                 self.ema_part,
                                 data_shape=[self.max_part,self.num_feat],
                                 cluster=tf.convert_to_tensor(cluster_info, dtype=tf.float32),
                                 const_shape = self.shape,
                                 mask=tf.convert_to_tensor(mask, dtype=tf.float32)).numpy()

        #print('parts',parts) # (5,200,4)
        #print(parts.shape) # (20, 200, 4)
        # print(parts*mask)
        
        end = time.time()
        print("Sampling Particles in {} Events ({} Seconds)".format(cond.shape[0],end - start))
        return parts,cluster_info


    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    cluster=None,
                    mask=None):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        
        batch_size = cond.shape[0] # 500
        t = tf.ones((batch_size,1))
        data_shape = np.concatenate(([batch_size],data_shape)) # [500   2]
        cond = tf.convert_to_tensor(cond, dtype=tf.float32) # Tensor("cond:0", shape=(500, 2), dtype=float32)
        init_x = self.prior_sde(data_shape) # Tensor("random_normal:0", shape=(500, 2), dtype=float32)

        print('mask',mask)
        print('init_x',init_x)

        if cluster is not None:
            init_x *= mask 

        x = init_x

        print('x_shape',x.shape)
        print('x',x)
        
        i = 0
        for  time_step in tf.range(self.num_steps-1, 0, delta=-1):
    
            i+=1
            #print('****************')
            batch_time_step = tf.ones((batch_size,1),dtype=tf.int32) * time_step

            z = tf.random.normal(x.shape,dtype=tf.float32)
            alpha = tf.gather(tf.sqrt(self.alphas_cumprod),batch_time_step)
            sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),batch_time_step)
            
            if cluster is None:
                score = model([x, batch_time_step,cond],training=False)
            else:
                score = model([x, batch_time_step,cluster,cond,mask],training=False)
                alpha = tf.reshape(alpha,self.shape)
                sigma = tf.reshape(sigma,self.shape)

            x_recon = alpha * x - sigma * score
            p1 = tf.reshape(tf.gather(self.posterior_mean_coef1,batch_time_step),const_shape)
            p2 = tf.reshape(tf.gather(self.posterior_mean_coef2,batch_time_step),const_shape)
            mean = p1*x_recon + p2*x
           
            log_var = tf.reshape(tf.gather(tf.math.log(self.posterior_variance),batch_time_step),const_shape)
            x = mean + tf.exp(0.5 * log_var) * z

            #print('x_final', x)

            '''
            score Tensor("model_1/dense_46/BiasAdd:0", shape=(500, 2), dtype=float32)
            self.posterior_mean_coef1 (512,)
            self.posterior_mean_coef2 (512,)
            mean Tensor("add_1:0", shape=(500, 2), dtype=float32)
            log_var Tensor("Reshape_2:0", shape=(500, 1), dtype=float32)
            x_final Tensor("add_2:0", shape=(500, 2), dtype=float32
            '''
            
        # The last step does not include any noise
        print('mean',mean)
        return mean        

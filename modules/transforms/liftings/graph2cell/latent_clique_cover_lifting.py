import networkx as nx
import torch_geometric
from toponetx.classes import CellComplex

import numpy as np
from scipy.special import gammaln, gamma
from scipy.sparse import csr_matrix
from scipy.special import gammaln

from modules.transforms.liftings.graph2cell.base import Graph2CellLifting



class LatentCliqueCoverSampler:
    """Latent clique cover model for network data corresponding to the
    Partial Observability Setting of the Random Clique Cover paper:
    http://proceedings.mlr.press/v115/williamson20a/williamson20a.pdf
    
    """
    def __init__(self, network, links, pie):
        self.alpha = 1
        self.sigma = 0.5
        self.lamb = 1000
        self.pie = pie

        self.network = network - np.diag(np.diag(network))
        self.links = links
        self.num_nodes = network.shape[0]
        self.num_links = len(links)
        self.Z = np.zeros((self.num_links, self.num_nodes))
        for i in range(self.num_links):
            self.Z[i,self.links[i][0]]=1
            self.Z[i,self.links[i][1]]=1
            
        self.K = self.Z.shape[0]

        # prior parameters
        self.alpha_params = [1.,1.]
        self.sigma_params = [1.,1.]
        self.pie_params = [1.,1.]
        
    def set_hyperpriors(self,alpha_params = None, sigma_params = None, pie_params = None):
        if alpha_params is not None:
            self.alpha_params = alpha_params
        if sigma_params is not None:
            self.sigma_params = sigma_params
        if pie_params is not None:
            self.pie_params = pie_params
            
    def sample(self, num_iters = 1000, num_sm = 10,dot_every=100,sample_hypers=True, do_gibbs = True, verbose = False):
        #Gibbs seems to matter here
        for iter in range(num_iters):
            if do_gibbs:
                self.gibbs()
                if verbose and iter%dot_every==0:
                    print('iter ',iter,', gibbs done.')
                
            if sample_hypers:
                self.sample_hypers()
                if verbose and iter%dot_every==0:
                    print('iter ',iter,', sample_hypers done.')

            for _ in range(num_sm):
                self.splitmerge()
                if verbose and iter%dot_every==0:
                    print('iter ',iter,', splitmerge done.')
            
            if iter%dot_every==0:
                print('iter ',iter,', K=',self.K)

                
    def log_lik(self,sigma=None, alpha=None, alpha_only = False):
        #same as full
        if sigma is None:
            sigma = self.sigma
        if alpha is None:
            alpha = self.alpha
        ll = self.num_nodes*np.log(alpha)
        if alpha_only is False:
            ll += self.num_nodes*(np.log(1-sigma) - gammaln(self.K+1-sigma))
        mk = np.sum(self.Z,0)
       
        c1 = gamma(2-sigma)*alpha
        for clique in range(1,self.K+1):
            log_gamma_ratio = gammaln(clique) - gammaln(clique+1-sigma)
            ll -= c1*np.exp(log_gamma_ratio)
        
        if alpha_only is False:
            for node in range(self.num_nodes):
                ll +=gammaln(mk[node]-sigma) + gammaln(self.K-mk[node]+1)
                
        return ll
                              
            
    
    def sample_hypers(self,step_size = 0.01):
        #same as full, but with sampling pie added in
        # mk = np.sum(self.Z,0)
        alpha_prop = self.alpha+step_size*np.random.randn()
        if alpha_prop>0:
            lp_ratio = (self.alpha_params[0]-1)*(np.log(alpha_prop)-np.log(self.alpha)) + self.alpha_params[1]*(self.alpha-alpha_prop)

            ll_new = self.log_lik(alpha = alpha_prop,alpha_only = True)
            ll_old = self.log_lik(alpha_only = True)
            lratio = ll_new - ll_old + lp_ratio
            r = np.log(np.random.rand())
            if r<lratio:
                self.alpha = alpha_prop
        sigma_prop = self.sigma+step_size*np.random.randn()
        if sigma_prop>0:
            if sigma_prop<1:
                ll_new = self.log_lik(sigma = sigma_prop)
                ll_old = self.log_lik()
                
                lp_ratio = (self.sigma_params[0]-1)*(np.log(sigma_prop)-np.log(self.sigma)) + (self.sigma_params[1]-1)*(np.log(1-sigma_prop)-np.log(1-self.sigma))
                lratio = ll_new - ll_old + lp_ratio
                r = np.log(np.random.rand())
                
                if r<lratio:
                    self.sigma = sigma_prop
                    
        pie_prop = self.pie + step_size*np.random.randn()
        if pie_prop>0:
            if pie_prop<1:
                ll_new = self.loglikZ(pie=pie_prop)
                ll_old = self.loglikZ()
                lp_ratio = (self.pie_params[0]-1)*(np.log(pie_prop)-np.log(self.pie))+(self.pie_params[1]-1)*(np.log(1-pie_prop)-np.log(1-self.pie))
                lratio = ll_new - ll_old + lp_ratio
                r = np.log(np.random.rand())
                if r<lratio:
                    self.pie = pie_prop
                
                     
    
    def gibbs(self):
        # empty_cliques = []
        mk = np.sum(self.Z,0)
        for node in range(self.num_nodes):
            for clique in range(self.K):
                if self.Z[clique,node]==1:
                    self.Z[clique,node] = 0
                    ll_0 = self.loglikZn(node)
                    self.Z[clique,node]=1
                    if not np.isinf(ll_0):
                        
                        ll_1 = self.loglikZn(node)
                        mk[node]-=1
                        if mk[node]==0:
                            raise ValueError('empty clique')
                            # pdb.set_trace() #shouldn't be possible, because we should break the ll check
                        #if it doesn't affect the network... sample from the prior
                        prior0 = (self.K-mk[node])/(self.K-self.sigma)
                        
                        prior1 = 1-prior0
                        
                        lp0 = np.log(prior0) + ll_0
                        lp1 = np.log(prior1) +ll_1
                        lp0 = lp0 - np.logaddexp(lp0,lp1)
                        r = np.log(np.random.rand())
                        if r<lp0:
                            self.Z[clique,node]=0
                            
                        else:
                            mk[node]+=1
                else:
                    self.Z[clique,node]=1
                    ll_1 = self.loglikZn(node)
                    self.Z[clique,node]=0
                    if not np.isinf(ll_1):
                        ll_0 = self.loglikZn(node)
                       
                        #if it doesn't affect the network... sample from the prior
                        prior0 = (self.K-mk[node])/(self.K-self.sigma)
                        
                        prior1 = 1-prior0
                        
                        lp0 = np.log(prior0) + ll_0
                        lp1 = np.log(prior1) +ll_1
                        lp1 = lp1 - np.logaddexp(lp0,lp1)
                        r = np.log(np.random.rand())
                        if r<lp1:
                            self.Z[clique,node]=1
                            mk[node]+=1
                    
    def loglikZ(self,Z=None,pie=None):
        if Z is None:
            Z = self.Z
        if pie is None:
            pie = self.pie
        cic = np.dot(Z.T,Z)
        cic = cic - np.diag(np.diag(cic))
        #check whether cic is ever zero, when network is 1
        zero_check = (1-np.minimum(cic,1))*self.network
        if np.sum(zero_check)==0:
            p0 = (1-pie)**cic
            p1 = 1-p0
            network_mask = self.network+1
            network_mask = np.triu(network_mask,1)-1
            #network_mask = np.triu(self.network,1)
            lp = np.sum(np.log(p0[np.where(network_mask==0)])) + np.sum(np.log(p1[np.where(network_mask==1)]))
            
        else:
            lp = -np.inf
        return lp
        
    def loglikZn(self,node,Z=None):
        if Z is None:
            Z = self.Z
        cic = np.dot(Z[:,node].T,Z)
        cic[node] = 0
        #check whether cic is ever zero, when network is 1
        zero_check = (1-np.minimum(cic,1))*self.network[node,:]
        if np.sum(zero_check)==0:
            p0 = (1-self.pie)**cic
            p1 = 1-p0
            lp = np.sum(np.log(p0[np.where(self.network[node,:]==0)])) + np.sum(np.log(p1[np.where(self.network[node,:]==1)]))
            
        else:
            lp = -np.inf
        return lp    
        
        
    def splitmerge(self):
        #pick an edge
        link_id = np.random.choice(self.num_links)
        r = np.random.rand()
        if r<0.5:
            sender = self.links[link_id][0]
            receiver = self.links[link_id][1]
        else:
            #randomizing because otherwise the distributions will be different in the split
            sender = self.links[link_id][1]
            receiver = self.links[link_id][0]
        #pick the first clique
        valid_cliques = np.where(self.Z[:,sender]==1)[0]
        clique_i = valid_cliques[np.random.choice(len(valid_cliques))]
        valid_cliques = np.where(self.Z[:,receiver]==1)[0]
        clique_j = valid_cliques[np.random.choice(len(valid_cliques))]

        if clique_i == clique_j:
            #propose split
            Z_prop = self.Z+0
            Z_prop = np.delete(Z_prop,clique_i,0)
            Z_prop = np.vstack((Z_prop, np.zeros((2,self.num_nodes))))
            
            lqsplit = 0
            lpsplit = 0
            
            mk = np.sum(Z_prop,0) 
            for node in range(self.num_nodes):#np.random.permutation(self.num_nodes):
                if self.Z[clique_i,node]==1:
                    if node == sender:
                        #must be 11 or 10
                        Z_prop[self.K-1,node]=1
                        
                        r = np.random.rand()
                        if r<0.5:
                            Z_prop[self.K,node]=1
                            #mk is one bigger, and K is one bigger, p1
                            lpsplit =lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+1-self.sigma)
                        else:
                            #mk is one bigger, and K is one bigger, p0
                            lpsplit = lpsplit + np.log(self.K+1-mk[node]-1) - np.log(self.K+1-self.sigma)
                        lqsplit -=np.log(2)
                        
                    elif node==receiver:
                        #must be 11 or 01
                        Z_prop[self.K,node]=1
                        r = np.random.rand()
                        if r<0.5:
                            Z_prop[self.K-1,node]=1
                            lpsplit =lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+1-self.sigma)
                        else:
                            lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K+1-self.sigma)
                        lqsplit -=np.log(2)
                    else:
                        r = np.random.rand()
                        if r<(1/3):
                            Z_prop[self.K-1,node]=1
                            #mk is one bigger, and K is one bigger, p0
                            lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K+1-self.sigma)
                        elif r<(2/3):
                            Z_prop[self.K,node]=1
                            lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K+1-self.sigma)
                        else:
                            Z_prop[self.K-1,node]=1
                            Z_prop[self.K,node]=1
                            #mk is one bigger, and K is one bigger, p1
                            lpsplit =lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K+1-self.sigma)
                        lqsplit -=np.log(3)
                else:
                    #mk is the same and K is one bigger, p0
                    lpsplit = lpsplit + np.log(self.K+1-mk[node]) - np.log(self.K+1-self.sigma)
                        
            #is the resulting proposal valid?
            
            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                ll_old = self.loglikZ()
                #then calculate the acceptance prob
                lqsplit = lqsplit -np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                #lqsplit =-np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                lqmerge = -np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]+np.sum(Z_prop[:,sender])) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]+np.sum(Z_prop[:,receiver]))
                
                lpsplit += np.log(self.lamb/(self.K+1))
                laccept = lpsplit -lqsplit+ lqmerge + ll_prop - ll_old
                r = np.log(np.random.rand())
            
                if r<laccept:
                    #pdb.set_trace()
                    #self.checksums
                    self.Z = Z_prop + 0
                    self.K+=1
                #self.checksums()
                
           
            
        else:
            #propose merge
            Z_sum = self.Z[clique_i,:]+self.Z[clique_j,:]
            Z_prop = self.Z+0
            Z_prop[clique_i]= np.minimum(Z_sum,1)
            Z_prop = np.delete(Z_prop,clique_j,0)
            ll_prop = self.loglikZ(Z_prop)
            if not np.isinf(ll_prop):
                #merge OK, proceed
                mk = np.sum(self.Z,0)-Z_sum 
                #calculate the backward probability
                num_affected = np.sum(Z_prop)
                if num_affected<2:
                    raise ValueError('num_affected<2')
                #lqsplit = -2*np.log(2) - (num_affected-2)*np.log(3)
                #OK now the merge probability
                lqmerge = -np.log(np.sum(self.Z[:,sender]))-np.log(np.sum(self.Z[:,receiver]))
                #lqsplit = lqsplit -np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]-self.Z[clique_j,sender]+1) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]-self.Z[clique_j,receiver]+1)
                lqsplit =-np.log(np.sum(self.Z[:,sender])-self.Z[clique_i,sender]-self.Z[clique_j,sender]+1) - np.log(np.sum(self.Z[:,receiver])-self.Z[clique_i,receiver]-self.Z[clique_j,receiver]+1)
                #lqsplit +=num_opt*np.log(0.5)
                
                lpsplit =0
                for node in range(self.num_nodes):
                    if Z_sum[node]==0:
                        #mk is the same, and K the same, p0
                        lpsplit = lpsplit + np.log(self.K-mk[node]) - np.log(self.K-self.sigma)
                    elif Z_sum[node]==1:
                        #mk is plus one, and K the same, p0
                        lpsplit = lpsplit + np.log(self.K-mk[node]-1) - np.log(self.K-self.sigma)
                    else:
                        #mk is plus one, and K the same, p2
                        lpsplit = lpsplit + np.log(mk[node]+1-self.sigma) - np.log(self.K-self.sigma)
                        
                lpmerge = np.log(self.K/self.lamb)
                ll_old = self.loglikZ()
                
                laccept = lpmerge -lpsplit +lqsplit - lqmerge + ll_prop - ll_old
                r = np.log(np.random.rand())
                
                if r<laccept:
                    self.Z = Z_prop+0
                    self.K-=1

                   

class LatentCliqueCoverLifting(Graph2CellLifting):
    r"""Lifts graphs to cell complexes by identifying the cycles as 2-cells.

    Parameters
    ----------
    max_cell_length : int, optional
        The maximum length of the cycles to be lifted. Default is None.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, pie: float = 0.8, it=100, warmup_it=10, **kwargs):
        super().__init__(**kwargs)
        self.pie = pie
        self.it = it
        self.warmup_it = warmup_it

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Finds the cycles of a graph and lifts them to 2-cells.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        net = torch_geometric.utils.to_dense_adj(data.edge_index)[0].numpy()
        for i in range(net.shape[0]):
            net[i, i] = 1
        links = data.edge_index.numpy().T

        mod = LatentCliqueCoverSampler(net, links, pie=self.pie)
        mod.sample(sample_hypers = False, num_iters=self.warmup_it, dot_every=1)
        mod.sample(sample_hypers = True, num_iters=self.it, dot_every=1)

        # extract to cells from membership matrix
        Z = mod.Z
        K = Z.shape[0]
        cells = []
        for n in range(K):
            nodes = np.where(Z[n, :] == 1)[0]
            cells.append(nodes)

        # make cell complex
        G = self._generate_graph_from_data(data)
        cc = CellComplex(G)

        # add 2-cells
        cc.add_cells_from(cells, rank=2)


# def poissonparams(K, alpha, sigma, c):
#     # vectorised for speed
#     ivec = np.arange(K, dtype=float)
#     lpp = (
#         np.log(alpha)
#         + gammaln(1.0 + c)
#         - gammaln(c + sigma)
#         + gammaln(ivec + c + sigma)
#         - gammaln(ivec + 1.0 + c)
#     )

#     pp = np.exp(lpp)

#     return pp


def sample_from_ibp(K, alpha, sigma, c):
    """
    samples from the random clique cover model using the three parameter ibp
    params
        K: number of random cliques
        alpha, sigma, c: ibp parameters
    returns
        a sparse matrix, compressed by rows, representing the clique membership matrix
        recover the adjacency matrix with min(Z'Z, 1)
    """
    # pp = poissonparams(K, alpha, sigma, c)
    ivec = np.arange(K, dtype=float)
    lpp = (
        np.log(alpha)
        + gammaln(1.0 + c)
        - gammaln(c + sigma)
        + gammaln(ivec + c + sigma)
        - gammaln(ivec + 1.0 + c)
    )
    pp = np.exp(lpp)
    new_nodes = np.random.poisson(pp)
    Ncols = new_nodes.sum()
    node_count = np.zeros(Ncols)

    # used to build sparse matrix, entries of each Zij=1
    colidx = []
    rowidx = []
    rightmost_node = 0

    # for each clique
    for n in range(K):
        # revisit each previously seen node
        for k in range(rightmost_node):
            prob_repeat = (node_count[k] - sigma) / (n + c)
            r = np.random.rand()
            if r < prob_repeat:
                rowidx.append(n)
                colidx.append(k)
                node_count[k] += 1

        for k in range(rightmost_node, rightmost_node + new_nodes[n]):
            rowidx.append(n)
            colidx.append(k)
            node_count[k] += 1

        rightmost_node += new_nodes[n]

    # build sparse matrix
    data = np.ones(len(rowidx), int)
    shape = (K, Ncols)
    Z = csr_matrix((data, (rowidx, colidx)), shape)

    return Z


if __name__ == "__main__":
    K, alpha, sigma, c= 30, 4, 0.7, 5
    Z = sample_from_ibp(K, alpha, sigma, c)

    adj = Z.transpose() @ Z
    g = nx.from_scipy_sparse_matrix(adj)
    for n in g.nodes():
        g.remove_edge(n, n)
        
    print("Number of links:", g.number_of_edges())
    print("Number of nodes:", g.number_of_nodes())

    # Transform to a torch geometric data object
    data = torch_geometric.utils.from_networkx(g)
    
    # Lift the topology to a cell complex
    lifting = LatentCliqueCoverLifting(piex=0.9, it=100)
    cell_complex = lifting.lift_topology(data)

    # Print the cell complex
    print(cell_complex)
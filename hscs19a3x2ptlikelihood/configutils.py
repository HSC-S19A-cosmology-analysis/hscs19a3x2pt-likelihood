import yaml
import os
import copy
import numpy as np

class config_class:
    def __init__(self, fname_or_dict):
        """
        Loads config file.
        config file has to have the following structure.
        - dataset
            - dirname
            - probes
            - covariance
            - samples
        - likelihood
            - name (str): name of likelihood
            - param
            - model
        - blind (bool): This is the analysis for a blind catalog or not.
        - output (str): dirname of output
        """
        if isinstance(fname_or_dict, dict):
            self.config = fname_or_dict
        elif isinstance(fname_or_dict, str):
            with open(fname_or_dict, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise NotImplemented
        
    def dump_config(self, fname):
        with open(fname, 'w') as f:
            yaml.dump(self.config, f, sort_keys=False)
        
    def get_dataset(self):
        """
        returns a dataset instance
        """
        dconf = copy.deepcopy(self.config['dataset'])
        from .datasetutils import dataset_class
        dataset = dataset_class(dconf)
        return dataset
        
    def get_like(self, verbose=True):
        """
        returns a likelihood instance
        """
        lconf = copy.deepcopy(self.config['likelihood'])
        dataset = self.get_dataset()
        
        if lconf['name'] == 'minimalbias':
            from .likelihood import minimalbias_likelihood_class
            like = minimalbias_likelihood_class(lconf, dataset, verbose=verbose)
            return like
        elif lconf['name'] == 'darkemu_x_hod':
            from .likelihood import darkemu_x_hod_likelihood_class
            like = darkemu_x_hod_likelihood_class(lconf, dataset, verbose=verbose)
            return like
        else:
            raise NotImplemented
            
    def get_MultiNest_base_name(self):
        return os.path.join(self.config['output'], 'mn-')
    
    def get_n_param_sampling(self):
        pconf = self.config['likelihood']['param']
        n = 0
        for name in pconf.keys():
            if pconf[name]['sample']:
                n+=1
        return n
    
    def get_names_param_sampling(self):
        pconf = self.config['likelihood']['param']
        names = []
        for name in pconf.keys():
            if pconf[name]['sample']:
                names.append(name)
        return names
    
    def get_names_param_full(self):
        pconf = self.config['likelihood']['param']
        names = list(pconf.keys())
        return names
        
    def get_dof_data(self):
        dataset = self.get_dataset()
        dof = dataset.probes.get_dof()
        return dof
    
    def get_dof_param(self):
        dof = np.sum([1 if i['sample'] else 0 for k,i in self.config['likelihood']['param'].items()])
        return dof
    
    def get_dof(self):
        ddof = self.get_dof_data()
        pdof = self.get_dof_param()
        return ddof, pdof

    def get_prior_cosmology(self, verbose=True):
        lconf = copy.deepcopy(self.config['likelihood'])
        dataset = self.get_dataset()
        
        from .likelihood import prior_cosmology_class
        like = prior_cosmology_class(lconf, dataset, verbose=verbose)
        return like
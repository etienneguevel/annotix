# Session Context

## User Prompts

### Prompt 1

when running train on ddp enabled there is a crash when loading the cached tensors on the different ranks ###[rank3]:     train_dataset, valid_dataset = make_datasets(                               
[rank3]:                                    ^^^^^^^^^^^^^^                               
[rank3]:   File "/raid/home/guevel/projects/annotix_all/annotix-ml/annotix_ml/graphtransf
/data/loaders.py", line 19, in make_datasets                                             
[rank3]:     train_dataset = Gr...

### Prompt 2

[rank3]:   File "/raid/home/guevel/projects/annotix_all/annotix-ml/.venv/lib/python3.12/site-packages/torch/utils/data/datalo
ader.py", line 1471, in _try_put_index                                                                                       
[rank3]:     index = self._next_index()                                                                                      
[rank3]:             ^^^^^^^^^^^^^^^^^^                                                                                   ...

### Prompt 3

[Request interrupted by user for tool use]


# Session Context

## User Prompts

### Prompt 1

The dataset making in the train.py script is a task that can be long, propose a way to make an option in a config to checkpoint it. If this option is activated then 1st it searches for files with the name and if it is not there then it makes the dataset and then saves it.

### Prompt 2

Put the logic to make the graph used before in a _ method of the dataset, when in distributed mode make the saving of the cache only true for the main rank


# Output files:

## Note: 
- Baseline, distance, argument features are between two events, so each line contains two events;
- Attribute features are for single event, and it can be used to train model to get, for example *Modality* feature for a event, 
and then be further used in caculating *Modality conflict* feature between two events;  

## baseline, distance, argument features for each event pairs within train docs:
> ACE_train_coref_features_3types_path_sim_reformat.json

## attribute features for each event within train docs:
> ACE_train_attribute_features_reformat.json

# Raw data: 
> /ACE05_data/train.oneie.json

> /ACE05_data/dev.oneie.json

> /ACE05_data/test.oneie.json


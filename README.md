# Textmorph repo

Repo for generating text according to editing and autocomplete style supervision.

## Example
Example runstring - 
```bash
export TEXTMORPH_DATA=/scr/nlp/squad_entailment_data
python docker.py -g 0 'python textmorph/edit_model/main.py configs/edit_model/edit_test.txt'
```
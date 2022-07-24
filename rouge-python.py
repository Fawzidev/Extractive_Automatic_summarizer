from pyrouge import Rouge155

r=Rouge155()

r.system_dir = 'mesres_centroid_synop'
r.model_dir = 'summaries-gold'
r.system_filename_pattern = 'Article(\d+).txt'
r.model_filename_pattern = 'Article#ID#.(\d+)(.[a-zA-Z]*)'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)
from os.path import dirname

####################################################
########          global configs           #########
####################################################

naming_scheme = 's2m2rf'
main_acc = 'test_acc'
firth_reg_col = 'firth_coeff'
PROJPATH = f'{dirname(dirname(__file__))}'
smry_tbls_dir = f'{PROJPATH}/summary'
paper_tbls_dir = f'{PROJPATH}/tables'

####################################################
########        csv2summ configs           #########
####################################################

reg_sources = ['test']
summ_cond_vars = ['source_dataset', 'target_dataset', 'n_ways', 'n_shots', 'n_aug']
results_csv_dir = f'{PROJPATH}/results'
# generating the path to the files to be read
# provided by a seperate .py file
specific_csv_fldrs = ['1_mini2CUB', '2_tiered2CUB', '3_tiered2tiered', '5ways']
deprecated_cols = []
prop_cols = ['n_shots', 'n_ways', 'source_dataset', 'target_dataset',
             'backbone_method', 'backbone_arch', 'split']
prop_cols = prop_cols + deprecated_cols
crn_cols = ['rng_seed', 'task_id']
dfltvals_dict = dict()

####################################################
########        summ2tables configs        #########
####################################################

table_sep_cols = ['target_dataset', 'source_dataset', 'n_ways', 'n_shots']
row_tree = ["n_ways", "n_shots"]
col_tree = ["source2target", "print_name"]

scale_percent = 100

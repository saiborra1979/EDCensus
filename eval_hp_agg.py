import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_list', nargs='+', help='Model classes to evaluate (xgboost lasso)')
args = parser.parse_args()
model_list = args.model_list
print(model_list)
assert isinstance(model_list, list)

model_list = ['gp_stacker']

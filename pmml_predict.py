# %%
# import sys
# !{sys.executable} -m pip install pypmml
# %%
# https://www.autodeploy.ai/
# https://github.com/autodeployai
# https://pypi.org/project/pypmml/
from pypmml import Model
import pandas as pd

# The model is from http://dmg.org/pmml/pmml_examples/KNIME_PMML_4.1_Examples/single_iris_dectree.xml
model = Model.load('library.pmml')
# %%
model.inputNames

# %%
model.predict({'number_months_available': 5,
 'total_checkouts': 451,
 'previous_quarter_checkouts': 231,
 'previous_month_checkouts': 125,
 'current_month_checkouts': 100,
 'total_collection_central': 75,
 'previous_quarter_collection_central': 25,
 'previous_month_collection_central': 42,
 'current_month_collection_central': 36,
 'total_collection_other': 75,
 'previous_quarter_collection_other': 76,
 'previous_month_collection_other': 82,
 'current_month_collection_other': 23})
# %%

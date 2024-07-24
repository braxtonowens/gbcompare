# The ACE Descriptor

## Unfortunate three-letter acronyms
ACE (Atomic Cluster Expansion)
ASE (Atomic Simulation Environment)

## ASE to XYZ
To describe the grain boundaries stored at gbcompare/gbs_ase.pickle you first need to turn the ASE object into a XYZ object by running ase_to_xyz.ipynb with a Python kernel. This will create a directory and fill it with the new XYZ objects stored as .xyz files.

## Describe GBs with ACE
Next in a julia environemnt you need to run get_ace.ipynb (see notebook for instructions). This will create a directory ace_txt_data with the ACE representation saved as .txt files.

## Machine learning with ACE
Lastly using gb_rep_ml
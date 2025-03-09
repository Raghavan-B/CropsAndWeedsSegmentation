from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    '''
    This function will return list of required libraries/packages for the project
    '''
    requirement_list:List[str] = []
    try:
        with open('requirements.txt','r') as file:
            ## Reading the packages from the file  
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                ## ignoring empty and -e lines
                if requirement and requirement!='-e .' and not requirement== '--extra-index-url https://download.pytorch.org/whl/cu118':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print('requirements.txt not found')

    return requirement_list

setup(
    name='CropsAndWeedsSegmentation',
    version='0.0.1',
    author='Raghavan B',
    packages=find_packages(),
    install_requires = get_requirements()
)


class FeatureSpec:
    def __init__(self,target:str , groups : dict[str , list[str]]):
        self.target = target
        self.groups  = groups


    def __str__(self):
        print('Target',self.target)
        print('-------------------------------')
        print('Feature A:' , self.groups['A'])
        print('######################################')
        print('Feature B:' , self.groups['B'])
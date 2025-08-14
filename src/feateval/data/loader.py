
class SpineGraphDataset():
    def __init__(self, rootA: str,rootB: str,GetObject):
        self.filesA = GetObject(rootA)
        self.filesB = GetObject(rootB)

    def len(self):  
        return {'A':len(self.filesA),'B':len(self.filesB)}

    def get(self, idx):  
        path = self.files[idx]
        objA = None
        objB = None
        if idx < len(self.filesA):
            objA = self.filesA[idx]
        if idx < len(self.filesB):
            objB = self.filesB[idx]
        
        return {'A': objA,'B':objB}





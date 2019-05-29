class point():
    def __init__(self):
        self.x=None
        self.y=None

    def add(self,i,j):
        self.x=i
        self.y=j

    def __repr__(self):
        return "u'<x:{},y:{}>".format(self.x,self.y)

def main():
    lst=[]
    for i in range(10):
        for j in range(10):
            temp = point()
            temp.add(i,j)
            lst.append(temp)

    print(lst)
    print(len(lst))

    clst = lst.copy()

    for i in lst:
        if i.x<5:
            clst.remove(i)
    
    print(lst)
    print(clst)
    print(len(lst))
    print(len(clst))
    return
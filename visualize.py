import matplotlib.pyplot as plt

class visualize:

 points_y=[]
 points_x=[]
 fig=plt.figure(figsize=[10,8])
 graph=fig.subplots()

 def __init__(self,title,y_label,x_label):
            self.graph.set_ylabel(y_label)
            self.graph.set_xlabel(x_label)
            self.graph.set_title(title)
            
 def addpoint_y(self,pointy):
         self.points_y.append(pointy)
         if (len(self.points_x) ==0):
            self.points_x=[1]
         else:
            self.points_x.append(len(self.points_x)+1)
         self.graph.plot(self.points_x,self.points_y,'bo')
         self.fig.show()
         plt.pause(0.0001)

            
    



a = visualize("cost function","cost","iterations")
for i in range(50):
           a.addpoint_y(i)

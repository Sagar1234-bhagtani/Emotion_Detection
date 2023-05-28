from collections import Counter

dict1={}

l=1
dict1[1]={1:"sagar"}
dict1[1]={1:"sagar"}
dict1[1]={1:"sagar"}
for i in range(1,10):
   l+=1
   dict1[1][l]="emotion"
   #print(dict1[i][l])

#print(dict1.get(1))
frequency = Counter(dict1.get(1).values())
print(frequency)
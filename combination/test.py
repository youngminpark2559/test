source=[0,1,1] # Success
# source=[1,0,1] # Success
# source=[1,1,0] # Fail, but why?
print("source",source)

result_li=[]
for i in range(2):
  for k in range(2):
    for j in range(2):
      print("i,k,j",i,k,j)
      print("source[i],source[k],source[j]",source[i],source[k],source[j])
      result_li.append([source[i],source[k],source[j]])
print("result_li",result_li)

# ================================================================================
# Test with source of [1, 0, 1]
# Above source creates combination of [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
# This is successful for the purpose

# source [1, 0, 1]
# i,k,j 0 0 0
# source[i],source[k],source[j] 1 1 1
# i,k,j 0 0 1
# source[i],source[k],source[j] 1 1 0
# i,k,j 0 1 0
# source[i],source[k],source[j] 1 0 1
# i,k,j 0 1 1
# source[i],source[k],source[j] 1 0 0
# i,k,j 1 0 0
# source[i],source[k],source[j] 0 1 1
# i,k,j 1 0 1
# source[i],source[k],source[j] 0 1 0
# i,k,j 1 1 0
# source[i],source[k],source[j] 0 0 1
# i,k,j 1 1 1
# source[i],source[k],source[j] 0 0 0
# result_li [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]]

# ================================================================================
# Test with source of [0, 1, 1]
# Above source creates combination of [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
# This is successful for the purpose

# source [0, 1, 1]
# i,k,j 0 0 0
# source[i],source[k],source[j] 0 0 0
# i,k,j 0 0 1
# source[i],source[k],source[j] 0 0 1
# i,k,j 0 1 0
# source[i],source[k],source[j] 0 1 0
# i,k,j 0 1 1
# source[i],source[k],source[j] 0 1 1
# i,k,j 1 0 0
# source[i],source[k],source[j] 1 0 0
# i,k,j 1 0 1
# source[i],source[k],source[j] 1 0 1
# i,k,j 1 1 0
# source[i],source[k],source[j] 1 1 0
# i,k,j 1 1 1
# source[i],source[k],source[j] 1 1 1
# result_li [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

# ================================================================================
# Test with source of [1, 1, 0]
# Above source creates [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
# This is failure for the purpose, but why did this fail?
# It's because the range of indexing is between 0 to 1 from source list
# So if the source list is [1,1,0], you are only able to use [1,1] from the list of [1,1,0]
# Or if you don't use source list [1,1,0], you also can use indexing number combination itself such as from "i,k,j 0 0 0" to "i,k,j 1 1 1"

# source [1, 1, 0]
# i,k,j 0 0 0
# source[i],source[k],source[j] 1 1 1
# i,k,j 0 0 1
# source[i],source[k],source[j] 1 1 1
# i,k,j 0 1 0
# source[i],source[k],source[j] 1 1 1
# i,k,j 0 1 1
# source[i],source[k],source[j] 1 1 1
# i,k,j 1 0 0
# source[i],source[k],source[j] 1 1 1
# i,k,j 1 0 1
# source[i],source[k],source[j] 1 1 1
# i,k,j 1 1 0
# source[i],source[k],source[j] 1 1 1
# i,k,j 1 1 1
# source[i],source[k],source[j] 1 1 1
# result_li [[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]


import pandas as pd
import random

# ================================================================================
pre_sale=pd.read_csv('/mnt/external_disk/Companies/B_Link/NHIS_OPEN_GJ_2017.CSV',encoding='utf8')
# print("pre_sale",pre_sale)
#         기준년도  가입자일련번호  성별코드  연령대코드(5세단위)  ...  치아마모증유무  제3대구치(사랑니)이상   치석   데이터공개일자
# 0       2017        1     1           13  ...      NaN           NaN  1.0  20181126
# 1       2017        2     2            8  ...      NaN           NaN  1.0  20181126

# print("pre_sale",pre_sale.shape)
# (1000000, 34)

# ================================================================================
number_for_divide_data=10

# ================================================================================
def chunks(l,n):
  for i in range(0,len(l),n):
    yield l[i:i+n]

# ================================================================================
one_group_size=int(pre_sale.shape[0]/number_for_divide_data)
# print("one_group_size",one_group_size)
# one_group_size 333333

# ================================================================================
indices_for_1mil=list(range(0,1000000))
random.shuffle(indices_for_1mil) # << shuffle before print or assignment
# print(indices_for_1mil)

# ================================================================================
list_temp=list(chunks(indices_for_1mil,one_group_size))
# print("list_temp",len(list_temp))
# print("list_temp",list_temp)

list_temp2=[]
for one_element in list_temp:
  if len(one_element)>3:
    list_temp2.append(one_element)

# print("list_temp2",list_temp2)
# print("list_temp2",len(list_temp2))
# 10

# ================================================================================
splited_data_list=[]

for i in range(number_for_divide_data):
  splited_data_list.append(pre_sale.iloc[list_temp2[i],:])

# print("splited_data_list",splited_data_list)
# print("splited_data_list",len(splited_data_list))
# 10

import random
subregion_txtpath='/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_subregions.txt'
with open(subregion_txtpath, 'r') as file:
        subregion_lines = file.readlines()
center_health=[]
for ii in range(len(subregion_lines)):
        if 'Health_' in subregion_lines[ii] and '0/0' in subregion_lines[ii] and '_bottom' in subregion_lines[ii]:
                center_health.append(subregion_lines[ii])
random.shuffle(center_health)
savetxtpath = '/disk3/wjr/dataset/nejm/shanxidataset/stage1_health_txt/selected_dcm_saomiao_900_health308_sick_592/train_bottomhealth413.txt'

for jj in range(413):
        with open(savetxtpath, 'a') as ff:
                ff.write((f"{center_health[jj]}"))

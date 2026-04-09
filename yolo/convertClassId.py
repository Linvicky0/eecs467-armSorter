'''
LabelStudio annotate template. Expand to different shapes if needed with alias filled in
<View>
  <Image name="image" value="$image"/>

  <RectangleLabels name="label" toName="image">
    <!-- small -->
    <Label value="small red cube" alias="2"/>
    <Label value="small orange cube" alias="4"/>
    <Label value="small yellow cube" alias="6"/>
    <Label value="small green cube" alias="8"/>
    <Label value="small blue cube" alias="10"/>
    <Label value="small purple cube" alias="12"/>

    <!-- large -->
    <Label value="large red cube" alias="14"/>
    <Label value="large orange cube" alias="16"/>
    <Label value="large yellow cube" alias="18"/>
    <Label value="large green cube" alias="20"/>
    <Label value="large blue cube" alias="22"/>
    <Label value="large purple cube" alias="24"/>
  </RectangleLabels>
</View>
'''

import argparse
import json
import glob
import sys 
import os 

parser = argparse.ArgumentParser(description="Resize images")
parser.add_argument("--annotated_dir", type=str,  required=True, help="Path to annotated dir")
args = parser.parse_args()

print("Input dir: ", args.annotated_dir)

conversionDict = {}

def conversionFunc(name):
    return str(int(name) - 1)

with open(f'{args.annotated_dir}/notes.json', 'r', encoding='utf-8') as file:
    try:
        notesJson = json.load(file)
    except Exception as e:
        print("can't open notes.json from input dir: ", args.annotated_dir)
        sys.exit(1)
    for item in notesJson["categories"]:
        inputId = str(item["id"])
        convertId = conversionFunc(item["name"])
        if convertId in conversionDict.values():
            print(f"value {convertId} in conversionDict; not a 1-to-1 mapping")
            print(conversionDict)
            sys.exit(1)
        conversionDict[inputId] = convertId

print("conversionDict: ", conversionDict)


rawImages = glob.glob(f"{args.annotated_dir}/images/*.png")
labelTxts = glob.glob(f"{args.annotated_dir}/labels/*.txt")
# print("rawImages: ", rawImages)
# print("labelTxts: ", labelTxts)
def convertIds():
    # convert id of each label.txt to the right class id
    for path in labelTxts:
        # filename = path.split('/')[-1]
        # output_file = filename.split('-')[-1]
        newContent = ''
        with open(path, 'r') as f:
            for line in f:
                contentList = line.strip().split()
                id = contentList[0]
                contentList[0] = conversionDict[id]
                line = ' '.join(contentList)
                line += '\n'
                newContent += line
        with open(path, 'w') as f:
            f.write(newContent)

# rename files in the same dir
def renameFiles():
    pngs  = glob.glob(f"{args.annotated_dir}/**/*.png")
    txts = glob.glob(f"{args.annotated_dir}/**/*.txt")
    for path in pngs + txts:
        dirpath = '/'.join(path.split('/')[0:-1]) # skip file name
        filename = path.split('/')[-1]
        if filename.startswith("frame"): # already the right filename 
            continue
        output_filename = filename.split('-')[-1]
        os.rename(f"{dirpath}/{filename}", f"{dirpath}/{output_filename}")
    

if __name__ == "__main__":
    convertIds()
    renameFiles()




      

import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import sys
import json
import numpy as np
import datetime
from similarity import *
import matplotlib.pyplot as plt
print(sys.argv)
class BoundingBox:
    ### Coordinates from top left corner
    
    def __init__(self):
        self.corners = np.array([[0,0],[0,0],[0,0],[0,0]])
        
    def center(self) -> np.array:
        result = np.array([0,0])
        for corner in self.corners:
            result += corner
            
        return result/len(self.corners)
    
    def lineHeight(self) -> float:
        minY = np.min(self.corners[:,1])
        maxY = np.max(self.corners[:,1])
        return maxY-minY
    
    def verticalDistanceTo(self, other) -> float:
        distance = np.linalg.norm(self.center()[1]-other.center()[1])
        return distance
        
    def isItVerticallyClose(self, other, nLines:float) -> bool:
        distance = self.verticalDistanceTo(other)
        reference = nLines*(self.lineHeight())   
        return distance <= reference
    
    def isInTheSameLine(self, other ) -> bool:
        return self.isItVerticallyClose(other,1)


    
class ExtractedField:
    keywords: list[str]
    name : str 
    extractedValue : str
    def __init__(self,name, keywords):
        self.name = name
        self.keywords = keywords
        self.extractedValue = None

class FuelOrder:
    name: str
    amount:str
class ExtractedFields:
    # fuelOrders: list[FuelOrder]
    # ship_to_id : ExtractedField
    BOL_number : ExtractedField
    # delivery_date: ExtractedField    
    # carriers :ExtractedField
    # terminal : ExtractedField
    # order_number : ExtractedField
    # city : ExtractedField
    # source : ExtractedField
    # supplier : ExtractedField
    # address_location : ExtractedField
    # delivery_address : ExtractedField
    # consignee_name     : ExtractedField    
    fields : list[ExtractedField]
    def __init__(self):
        
        BOL_number = ExtractedField("BOL_number",["BOL","Bill of Lading", "Invoice"])
        delivery_date = ExtractedField("delivery_date",["Delivery Date","Arrive", "Date Shipped"])

        self.fields = [BOL_number,delivery_date]
    

class TextField:
    text : str
    boundingBox : BoundingBox
 
class ScannedDocumentAnalysis:
    textFiels: TextField
    

#testb = BoundingBox()
#testb.corners = np.array([[20,20],[100,20],[100,40],[20,40]])

#testOther = BoundingBox()
#testOther.corners = np.array([[20+10,80],[100+10,80],[100+10,100],[20+10,100]])
#print(testb.center())
#print(testb.lineHeight())
#print(testOther.center())
#print(testOther.lineHeight())

#print(testb.isItVerticallyClose(testOther,3))
#quit()



# Set the values of your computer vision endpoint and computer vision key
# as environment variables:
try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

# Create an Image Analysis client
client = ImageAnalysisClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(key)
)



        
        


image_data = None
inputFileName = sys.argv[1]+".png"
with open(inputFileName, "rb") as f:
    image_data = f.read()


# Get a caption for the image. This will be a synchronously (blocking) call.
#result = client.analyze_from_url(
#    image_url="https://learn.microsoft.com/azure/ai-services/computer-vision/media/quickstarts/presentation.png",
result = client.analyze(
    image_data=image_data,
    visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
    gender_neutral_caption=True,  # Optional (default is False)
)

print("Image analysis results:")

#outputFileName = sys.argv[1]+".json"
#with open(outputFileName, "w") as f:
#    #f.write(str(result))
#    json.dump(result.read.blocks, f,ensure_ascii=False, indent=4)

# Print caption results to the console
print(" Caption:")
if result.caption is not None:
    print(f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}")

# Print text (OCR) analysis results to the console
print(" Read:")
#print(result)


if result.read is not None:
    for line in result.read.blocks[0].lines:
        print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

fieldsToExtract = ExtractedFields()
similarityThreshold = 0.7
if result.read is not None:
    for line in result.read.blocks[0].lines:
        for field in fieldsToExtract.fields:
                similarityLevelsFound = similarities(field.keywords, line.text)
                similarityFoundQ, index = isItSimilar(similarityLevelsFound,similarityThreshold) 
                if similarityFoundQ:
                    field.extractedValue = line.text
                    print("Similarity found!")
                    print("line: ", line.text)
                    print("field: ", field.name)
                    print("Similarity: ", similarityLevelsFound[index])
                    #break

                    
        #print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        #for word in line.words:
        #    print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")


img = plt.imread(inputFileName)
plt.imshow(img)


plt.show()


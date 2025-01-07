import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
import sys
import json
import numpy as np
import datetime

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
    
    def isItClose(self, other, nLines:float) -> bool:
        distance = np.linalg.norm(self.center()-other.center())
        reference = nLines*(self.lineHeight())   
        return distance <= reference

class FuelOrder:
    name: str
    amount:str
class ExtractedFields:
    fuelOrders: list[FuelOrder]
    ship_to_id : str
    BOL_number : str
    delivery_date: datetime    
    carriers :str
    terminal : str
    order_number : str
    city : str
    source : str
    supplier : str
    address_location : str
    delivery_address : str
    consignee_name     : str    
            
        
testb = BoundingBox()
testb.corners = np.array([[20,20],[100,20],[100,40],[20,40]])

testOther = BoundingBox()
testOther.corners = np.array([[20,80],[100,80],[100,100],[20,100]])
print(testb.center())
print(testb.lineHeight())
print(testOther.center())
print(testOther.lineHeight())

print(testb.isItClose(testOther,3))
quit()



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
print(result)
if result.read is not None:
    for line in result.read.blocks[0].lines:
        print(f"   Line: '{line.text}', Bounding box {line.bounding_polygon}")
        for word in line.words:
            print(f"     Word: '{word.text}', Bounding polygon {word.bounding_polygon}, Confidence {word.confidence:.4f}")

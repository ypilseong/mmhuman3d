import numpy
from json import JSONEncoder
import json

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

x = numpy.load('/home/pilseong/OpenMMlab/mmhuman3d/demo_result/inference_result_multi.npz', allow_pickle=True)

# Serialization
numpyData = {"arrayOne": x} #, "arrayTwo": x['person_id']}
print("serialize NumPy array into JSON and write into a file")
with open("numpyData.json", "w") as write_file:
    json.dump(numpyData, write_file, cls=NumpyArrayEncoder)
print("Done writing serialized NumPy array into file")

# Deserialization
#print("Started Reading JSON file")
#with open("numpyData.json", "r") as read_file:
#    print("Converting JSON encoded data into Numpy array")
#    decodedArray = json.load(read_file)
#
 #   finalNumpyArrayOne = numpy.asarray(decodedArray["arrayOne"])
  #  print("NumPy Array One")
   # print(finalNumpyArrayOne)
    #finalNumpyArrayTwo = numpy.asarray(decodedArray["arrayTwo"])
    #print("NumPy Array Two")
    #print(finalNumpyArrayTwo)
import argparse 
from predict import predict

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--image", required=True, 
	help="path to input image for the model")
parser.add_argument("-w", "--writepath", 
	help="where to write the result")	
parser.add_argument("-m", "--model", type=str, default="model/model.h5",
	help="path that contains the wieghts for the model to use")

args = vars(parser.parse_args())

modelpath = args["model"]
imagepath = args["image"]
writepath = args["writepath"]
if not writepath :
	writepath = imagepath

predict(modelpath, imagepath, writepath)
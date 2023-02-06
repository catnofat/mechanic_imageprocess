import os
import cv2
import numpy as np
import glob
import csv

imgfiles = glob.glob('.\\1\\*.png')
alldot=[]
#areafilter reference : https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python

def areaFilter(minArea, inputImage):
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage
a = 0
for imagepath in imgfiles:
    a+=1
    dotlist = []

    #find yellow images-> white, else : black
    img = cv2.imread(imagepath)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([22, 130, 100])
    upper_yellow = np.array([40, 255, 230])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    #noise eliminate
    minArea = 30    # noise elimination magnitude
    binaryImage = areaFilter(minArea, mask) 

    # manual eliminate(yellow in background->black)
    for x in range(739,1050):
        for y in range(130, 230):
            binaryImage[y,x] = 0
    
    # find white an save x,y. 3pixe
    y=0
    while y < 957:
        y+=3    # not find +3 y pixel, find +23 y pixel.
        x= 0
        if(y>957):
            break
        while x < 1050:
            if(binaryImage[y,x]==255):
                dotlist.append([x,y])
                y+=15
                break   # find-> then go to next y(+20pixel)
            x+=3    #try every 3 pixel
    
    #find center
    modidotlist = []


    for i in range(len(dotlist)):
        startx = dotlist[i][0]-7
        starty = dotlist[i][1]-7
        centerx = 0
        centery = 0
        count = 0  
        for j in range(1,15):
            for k in range(1,15):
                newx = startx + j
                newy = starty + k
                if(binaryImage[newy, newx] == 255):
                        centerx += newx
                        centery += newy
                        count+=1
        centerx = centerx/count
        centery = centery/count 
        modidotlist.append([centerx, centery])


    # draw box
    print(len(dotlist))
    for i in range(len(dotlist)):
        cv2.rectangle(binaryImage, (int(modidotlist[i][0]-7), int(modidotlist[i][1]-7), 14, 14), 255, 1)

    alldot.append(modidotlist)
    print(alldot)


    #if len(dotlist) !=9:
    #cv2.imshow('bmask', binaryImage)
    cv2.imwrite('cam(1)'+str(a)+'.png',binaryImage)
    #cv2.imshow('mask', mask)
    #cv2.waitKey(0)


npalldot = np.array(alldot)
np.save('List2.npy', npalldot)

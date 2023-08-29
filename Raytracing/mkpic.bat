@echo off
g++ ./ppm.cpp -o ppm.exe && ppm > image.ppm
echo "Creating Image.ppm Finished"
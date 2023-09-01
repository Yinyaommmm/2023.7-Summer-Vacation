@echo off
g++ ./main.cpp -o main.exe && main.exe > image.ppm
echo "Creating main.ppm Finished"
## IntelEdgeProject1

## Run the app
    python3.5 main.py -d MYRIAD -i resources/Pedestrian_Detect_2_1_1.mp4 -m my_model.xml -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

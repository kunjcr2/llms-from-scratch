######################################################
# # EXACTLY LIKE WE DO IN MULTITHREAD.
# import multiprocessing
# import time

# def do_something(sec):
#     print(f"Sleeping for {sec} sec")
#     time.sleep(1)
#     print("I'm up")

# if __name__ == "__main__":
#     start = time.time()

#     process = []
#     # Total of 10 processes together just like Threading
#     for _ in range(10):
#         p = multiprocessing.Process(target=do_something, args=[1.5])
#         p.start()
#         process.append(p)
    
#     # joining them to the main process just like Threading.
#     for p in process:
#         p.join()

#     finish = time.time()
#     print(f"Slept for {round(finish-start, 2)} seconds")
######################################################

# # Using PoolExecutor to use multi processing
# # SAME AS MULTITHREAD !
# import concurrent.futures
# import time

# def do_something(sec: float) -> str:
#     print(f"Sleeping for {sec} sec")
#     time.sleep(sec)
#     return "I'm up"

# def main():
#     start = time.time()

#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         secs = [5,4,3,2,1]
#         processes = [executor.submit(do_something, sec) for sec in secs]
#         for process in processes:
#             print(process.result())

#     finish = time.time()
#     print(f"Slept for {round(finish - start, 2)} seconds")

# if __name__ == "__main__":
    # main()
######################################################

# We do some stuff on the images.
import time
import concurrent.futures
from PIL import Image, ImageFilter  # Image manager library

img_names = [
    'photo-1516117172878-fd2c41f4a759.jpg',
    'photo-1532009324734-20a7a5813719.jpg',
    'photo-1524429656589-6633a470097c.jpg',
    'photo-1530224264768-7ff8c1789d79.jpg',
    'photo-1564135624576-c5c88640f235.jpg',
    'photo-1541698444083-023c97d3f4b6.jpg',
    'photo-1522364723953-452d3431c267.jpg',
    'photo-1513938709626-033611b8cc03.jpg',
    'photo-1507143550189-fed454f93097.jpg',
    'photo-1493976040374-85c8e12f0c0e.jpg',
    'photo-1504198453319-5ce911bafcde.jpg',
    'photo-1530122037265-a5f1f91d3b99.jpg',
    'photo-1516972810927-80185027ca84.jpg',
    'photo-1550439062-609e1531270e.jpg',
    'photo-1549692520-acc6669e2f0c.jpg'
]

size = (1200, 1200)

def process_image(img_name):
    img = Image.open(f'./assets/{img_name}')
    img = img.filter(ImageFilter.GaussianBlur(15))
    img.thumbnail(size)
    img.save(f'./assets/processed/{img_name}')
    print(f'{img_name} was processed...')

def main():
    t1 = time.time()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_image, img_names))  # force execution

    t2 = time.time()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == "__main__":
    main()

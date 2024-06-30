import cv2
import numpy as np
import os

# Ukuran gambar yang akan disimpan
image_x, image_y = 64, 64

# Fungsi ini membuat folder untuk set pelatihan dan set pengujian jika belum ada.
# Folder dibuat berdasarkan nama gestur yang diberikan sebagai argumen.
def create_folder(folder_name):
    if not os.path.exists('./mydata/training_set/' + folder_name):
        os.mkdir('./mydata/training_set/' + folder_name)
    if not os.path.exists('./mydata/test_set/' + folder_name):
        os.mkdir('./mydata/test_set/' + folder_name)
        
# Fungsi ini digunakan untuk menangkap gambar dari kamera, memprosesnya, 
# dan menyimpannya ke dalam folder yang sesuai untuk pelatihan dan pengujian.
def capture_images(ges_name):
    create_folder(str(ges_name))
    
    cam = cv2.VideoCapture(2)
    cv2.namedWindow("test")

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    listImage = [1, 2, 3, 4, 5]

    for loop in listImage:
        while True:
            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            # Membuat kotak persegi panjang di sekitar daerah pengambilan gambar
            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8, shift=0)

            # Ambil bagian yang dipilih untuk gambar (crop)
            imcrop = img[102:298, 427:623]

            # Ubah citra ke grayscale
            gray = cv2.cvtColor(imcrop, cv2.COLOR_BGR2GRAY)

            # Lakukan Otsu Thresholding
            ret, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Resize mask dan simpan gambar
            save_img = cv2.resize(mask, (image_x, image_y))

            # Menampilkan jumlah gambar yang telah diambil
            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))

            # Menampilkan frame dan hasil pemrosesan
            cv2.imshow("test", frame)
            cv2.imshow("mask", mask)

            # Menekan 'c' untuk menangkap gambar
            if cv2.waitKey(1) == ord('c'):
                if t_counter <= 350:
                    img_name = "./mydata/training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1

                # Menyimpan gambar dalam set pengujian setelah mencapai batas tertentu
                if t_counter > 350 and t_counter <= 400:
                    img_name = "./mydata/test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    if test_set_image_name > 250:
                        break

                t_counter += 1
                if t_counter == 401:
                    t_counter = 1
                img_counter += 1

            # Menekan 'ESC' untuk keluar
            elif cv2.waitKey(1) == 27:
                break

        if test_set_image_name > 250:
            break

    cam.release()
    cv2.destroyAllWindows()

# Meminta pengguna memasukkan nama gestur
ges_name = input("Enter gesture name: ")
capture_images(ges_name)

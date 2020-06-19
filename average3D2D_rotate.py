def average3d_to_2d_cv2():
    import cv2
    import easygui
    import mrcfile
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.ndimage as ndimage

    def nothing(x):
        pass

    def check_arg():
        import argparse

        parser = argparse.ArgumentParser(description='2D projection of the 3D volume with rotations')
        parser.add_argument("--i", type=str, help="Input MRC volume file")
        parser.add_argument("--box_size", type=int, default=100, help="Desired box size for the calculations")
        parser.add_argument("--win_size", type=int, default=500, help="Preview window size")
        parser.add_argument("--norm", type=bool, default=True, help="Normalize projection between 0 and 1")
        parser.add_argument("--only_resize", type=bool, default=False, help="Save only resized map and quit")

        args = parser.parse_args()
        mrcs_file = args.i
        if mrcs_file == None:
            mrcs_file = easygui.fileopenbox()
        box_size = args.box_size
        win_size = args.win_size
        norm = args.norm
        only_resize = args.only_resize

        if box_size % 2 != 0:
            print('Box size has to be even!')
            quit()

        return mrcs_file, box_size, win_size, norm, only_resize

    def resize_3d(volume, new_size, only_resize):

        if new_size % 2 != 0:
            print('Box size has to be even!')
            quit()

        pixel_size = volume.voxel_size.x
        volume = volume.data
        original_size = volume.shape

        fft = np.fft.fftn(volume)
        fft_shift = np.fft.fftshift(fft)

        # cut this part of the fft
        x1, x2 = int((volume.shape[0] - new_size) / 2), volume.shape[0] - int((volume.shape[0] - new_size) / 2)

        fft_shift_new = fft_shift[x1:x2, x1:x2, x1:x2]

        # spherical mask
        lx, ly, lz = fft_shift_new.shape
        X, Y, Z = np.ogrid[0:lx, 0:ly, 0:lz]
        dist_from_center = np.sqrt((X - lx / 2) ** 2 + (Y - ly / 2) ** 2 + (Z - lz / 2) ** 2)
        mask = dist_from_center <= lx / 2
        fft_shift_new[~mask] = 0

        fft_new = np.fft.ifftshift(fft_shift_new)
        new = np.fft.ifftn(fft_new)

        # save resized map?
        if only_resize:
            try:
                new_file = mrcfile.open('fft.mrc', mode='r+')
            except:
                mrcfile.new('fft.mrc')
                new_file = mrcfile.open('fft.mrc', mode='r+')

            new_file.voxel_size = original_size[0] / lx * pixel_size
            new_file.set_data(new.real.astype(np.float16))
            new_file.close()
            print('Resized map saved!')
            quit()

        print('New box size: {}'.format(new.shape))

        #Real? abs? both work but give slighly different results. Might have to be fixed with proper in the future
        return np.abs(new)

    def rotate(mrcs_file, angle, rotaxes_):
        mrcs_file = ndimage.interpolation.rotate(mrcs_file, angle, reshape=False, axes=rotaxes_)
        return mrcs_file

    def project(volume, value_matrix, value_matrix_old, win_size, normalize):

        if rotaxes == 0:
            rotaxes_ = [0, 1]
        elif rotaxes == 1:
            rotaxes_ = [1, 2]
        elif rotaxes == 2:
            rotaxes_ = [0, 2]

        mrcs_file = volume

        # Do calulations only when sliders are changed
        if value_matrix != value_matrix_old:
            mrcs_file = rotate(mrcs_file, angle, rotaxes_)

            if axis == 1:
                average = np.zeros((z, y))
                for i in range(limit_low, limit_high):
                    img = mrcs_file[:, i, :]
                    average += img

            elif axis == 0:
                average = np.zeros((z, y))
                for i in range(limit_low, limit_high):
                    img = mrcs_file[i, :, :]
                    average += img

            elif axis == 2:
                average = np.zeros((z, y))
                for i in range(limit_low, limit_high):
                    img = mrcs_file[:, :, i]
                    average += img

            # Normalization of Average between 0 and 1
            if normalize:
                average = (average - np.min(average)) / (np.max(average) - np.min(average))

            average = cv2.resize(average, (win_size, win_size))
            cv2.imshow('Volume', average)


        cv2.namedWindow('Volume', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Volume', win_size, win_size)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            quit(1)

        # Save projection to file
        elif k == 32:
            name = 'img_{}.png'.format(np.random.randint(0, 1000, 1)[0])
            plt.imsave(name, average, cmap='gray')
            print('Image saved!: {}'.format(name))

    # here it starts
    mrcs_file, box_size, win_size, norm, only_resize = check_arg()

    with mrcfile.open(mrcs_file) as volume:

        mrcs_file = volume
        mrcs_file = resize_3d(mrcs_file, box_size, only_resize)
        z, x, y = mrcs_file.data.shape


        cv2.namedWindow('Volume')
        cv2.namedWindow('Volume', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Volume', win_size, win_size)
        cv2.createTrackbar('Axis', 'Volume', 0, 2, nothing)
        cv2.createTrackbar('Lower limit', 'Volume', 0, x, nothing)
        cv2.createTrackbar('Higer limit', 'Volume', 0, x, nothing)
        cv2.createTrackbar('rotaxes', 'Volume', 0, 2, nothing)
        cv2.createTrackbar('angle', 'Volume', 0, 360, nothing)

        axis = cv2.getTrackbarPos('Axis', 'Volume')
        limit_low = cv2.getTrackbarPos('Lower limit', 'Volume')
        limit_high = cv2.getTrackbarPos('Higer limit', 'Volume')
        rotaxes = cv2.getTrackbarPos('rotaxes', 'Volume')
        angle = cv2.getTrackbarPos('angle', 'Volume')

        value_matrix_old = [axis, limit_low, limit_high, angle]

        while (1):
            axis = cv2.getTrackbarPos('Axis', 'Volume')
            limit_low = cv2.getTrackbarPos('Lower limit', 'Volume')
            limit_high = cv2.getTrackbarPos('Higer limit', 'Volume')
            angle = cv2.getTrackbarPos('angle', 'Volume')
            rotaxes = cv2.getTrackbarPos('rotaxes', 'Volume')
            value_matrix = [axis, limit_low, limit_high, angle]

            project(mrcs_file, value_matrix, value_matrix_old, win_size, norm)
            value_matrix_old = value_matrix


if __name__ == "__main__":
    average3d_to_2d_cv2()

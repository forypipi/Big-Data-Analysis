# %%
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from PIL import Image
import os
%matplotlib inline
from matplotlib import pyplot as plt

# import gui
import ipywidgets as widgets
from IPython.display import display

# %%
def loadDcm(dcm_path: Path):
    """
    load dicom files from dcm_path(folder), return (ndarray, sitk.Image)
    """
    # 使用SimpleITK读取DICOM文件夹
    series_reader = sitk.ImageSeriesReader()
    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    dicom_series = series_reader.GetGDCMSeriesFileNames(str(dcm_path))
    series_reader.SetFileNames(dicom_series)
    ct_image = series_reader.Execute()

    # 将SimpleITK图像转换为NumPy数组
    ct_ndarray = sitk.GetArrayFromImage(ct_image)

    return ct_ndarray, ct_image

# %%
def laodMask(mask_path: Path):
    """
    load png files from mask_path(folder), return (ndarray)
    """

    # 获取文件夹中所有PNG文件的文件名
    mask_path = str(mask_path)
    png_files = sorted([f for f in os.listdir(mask_path) if f.endswith(".png")])

    # 通过循环读取PNG文件序列并存储在列表中
    mask_sequence = []
    for png_file in png_files:
        png_path = os.path.join(mask_path, png_file)
        image = Image.open(png_path)
        mask_sequence.append(image)

    masks = np.stack([np.array(image) for image in mask_sequence[::-1]])
    return masks

# %%
class MultiImageDisplay(object):

    def __init__(
        self,
        image_list,
        axis=0,
        shared_slider=False,
        title_list=None,
        window_level_list=None,
        intensity_slider_range_percentile=[2, 98],
        figure_size=(10, 8),
        horizontal=True,
    ):

        self.image_list, self.horizontal, self.figure_size = image_list, horizontal, figure_size

        self.npa_list, wl_range, wl_init = self.get_window_level_numpy_array(
            image_list, window_level_list, intensity_slider_range_percentile
        )
        if title_list:
            if len(image_list) != len(title_list):
                raise ValueError("Title list and image list lengths do not match")
            self.title_list = list(title_list)
        else:
            self.title_list = [""] * len(image_list)

        # Our dynamic slice, based on the axis the user specifies
        self.slc = [slice(None)] * 3
        self.axis = axis

        ui = self.create_ui(shared_slider, wl_range, wl_init)
        display(ui)

        if len(image_list) == 1:
            self.axes = [self.axes]

        # Display the data and the controls, first time we display the image is outside the "update_display" method
        # as that method relies on the previous zoom factor which doesn't exist yet.
        self.update_display()

    def create_ui(self, shared_slider, wl_range, wl_init):
        # Create the active UI components. Height and width are specified in 'em' units. This is
        # a html size specification, size relative to current font size.

        if shared_slider:
            # Validate that all the images have the same size along the axis which we scroll through
            sz = self.npa_list[0].shape[self.axis]
            for npa in self.npa_list:
                if npa.shape[self.axis] != sz:
                    raise ValueError(
                        "Not all images have the same size along the specified axis, cannot share slider."
                    )

            slider = widgets.IntSlider(
                description="image slice:",
                min=0,
                max=sz - 1,
                step=1,
                value=int((sz - 1) / 2),
                width="20em",
            )
            slider.observe(self.on_slice_slider_value_change, names="value")
            self.slider_list = [slider] * len(self.npa_list)
            slicer_box = widgets.Box(padding=7, children=[slider])
        else:
            self.slider_list = []
            for npa in self.npa_list:
                slider = widgets.IntSlider(
                    description="image slice:",
                    min=0,
                    max=npa.shape[self.axis] - 1,
                    step=1,
                    value=int((npa.shape[self.axis] - 1) / 2),
                    width="20em",
                )
                slider.observe(self.on_slice_slider_value_change, names="value")
                self.slider_list.append(slider)
            slicer_box = widgets.Box(padding=7, children=self.slider_list)
        self.wl_list = []
        # Each image has a window-level slider, but it is disabled if the image
        # is a color image len(npa.shape)==4 . This allows us to display both
        # color and grayscale images in the same UI while retaining a reasonable
        # layout for the sliders.
        for r_values, i_values, npa in zip(wl_range, wl_init, self.npa_list):
            wl_range_slider = widgets.IntRangeSlider(
                description="intensity:",
                min=r_values[0],
                max=r_values[1],
                step=1,
                value=[i_values[0], i_values[1]],
                width="20em",
                disabled=len(npa.shape) == 4,
            )
            wl_range_slider.observe(self.on_wl_slider_value_change, names="value")
            self.wl_list.append(wl_range_slider)
        wl_box = widgets.Box(padding=7, children=self.wl_list)
        return widgets.VBox(children=[slicer_box, wl_box])

    def get_window_level_numpy_array(
        self, image_list, window_level_list, intensity_slider_range_percentile
    ):
        # Using GetArray and not GetArrayView because we don't keep references
        # to the original images. If they are deleted outside the view would become
        # invalid, so we use a copy which guarantees that the GUI is consistent.
        npa_list = list(map(sitk.GetArrayFromImage, image_list))

        wl_range = []
        wl_init = []
        # We need to iterate over the images because they can be a mix of
        # grayscale and color images. If they are color we set the wl_range
        # to [0,255] and the wl_init is equal, ignoring the window_level_list
        # entry.
        for i, npa in enumerate(npa_list):
            if len(npa.shape) == 4:  # color image
                wl_range.append((0, 255))
                wl_init.append((0, 255))
                # ignore any window_level_list entry
            else:
                # We don't necessarily take the minimum/maximum values, just in case there are outliers
                # user can specify how much to take off from top and bottom.
                min_max = np.percentile(
                    npa.flatten(), intensity_slider_range_percentile
                )
                wl_range.append((min_max[0], min_max[1]))
                if not window_level_list:  # No list was given.
                    wl_init.append(wl_range[-1])
                else:
                    wl = window_level_list[i]
                    if wl:
                        wl_init.append((wl[1] - wl[0] / 2.0, wl[1] + wl[0] / 2.0))
                    else:  # We have a list, but for this image the entry was left empty: []
                        wl_init.append(wl_range[-1])
        return (npa_list, wl_range, wl_init)

    def on_slice_slider_value_change(self, change):
        self.update_display()

    def on_wl_slider_value_change(self, change):
        self.update_display()

    def update_display(self):

        # 创建新的子图对象
        col_num, row_num = (len(self.image_list), 1) if self.horizontal else (1, len(self.image_list))
        self.fig, self.axes = plt.subplots(row_num, col_num, figsize=self.figure_size)

        # Draw the image(s)
        for ax, npa, title, slider, wl_slider, in zip(
            self.axes, self.npa_list, self.title_list, self.slider_list, self.wl_list,
        ):
            # We want to keep the zoom factor which was set prior to display, so we log it before
            # clearing the axes.

            self.slc[self.axis] = slice(slider.value, slider.value + 1)
            # Need to use squeeze to collapse degenerate dimension (e.g. RGB image size 124 124 1 3)
            ax.imshow(
                np.squeeze(npa[tuple(self.slc)]),
                cmap=plt.cm.Greys_r,
                vmin=wl_slider.value[0],
                vmax=wl_slider.value[1],
            )
            ax.set_title(f"{title} {slider.value}-th")
            ax.set_axis_off()


        self.fig.canvas.draw_idle()
        plt.tight_layout()

# %%
def Mydisplay(ct_Image, mask_Image):

    ct_Image = sitk.Cast(
        sitk.IntensityWindowing(
            ct_Image,
            windowMinimum=-1000,
            windowMaximum=170,
            outputMinimum=0.0,
            outputMaximum=255.0,
            ),
        sitk.sitkUInt8,
    )
    
    half_overlap = sitk.LabelOverlay(
            image=ct_Image,
            labelImage=mask_Image,
            opacity=0.3,
            backgroundValue=0,
        )

    zeros = sitk.Image(ct_Image.GetSize(), ct_Image.GetPixelID())
    zeros.CopyInformation(ct_Image)

    half_ct = sitk.LabelOverlay(
            image=ct_Image,
            labelImage=zeros,
            opacity=0.3,
            backgroundValue=0,
        )
    
    MultiImageDisplay(
        image_list=[half_ct, mask_Image, sitk.LabelToRGB(mask_Image), half_overlap],
        title_list=["image", "raw segmentation labels", "segmentation labels in color", "fuse image"],
        figure_size=(18, 18),
        shared_slider=True,
        axis=0,
    )


# %%
if __name__=="__main__":
    root = Path(r"C:\Users\orfu\Desktop\MyData\BaiduSyncdisk\研究方向\小样本学习\数据集\CHAOS\CHAOS_Train_Sets\Train_Sets\CT")
    for patient in os.listdir(root)[:1]:
        dicom_folder = root / patient / "DICOM_anon"
        mask_folder = root / patient / "Ground"
        
        ct_ndarray, ct_image = loadDcm(dicom_folder)
        mask_ndarray = laodMask(mask_folder).astype(int)
        mask_image = sitk.GetImageFromArray(mask_ndarray)
        mask_image.CopyInformation(ct_image)

        Mydisplay(ct_image, mask_image)
        print(f"patient: {patient}, \tct shape: {ct_ndarray.shape}, \tmask shape: {mask_ndarray.shape}")



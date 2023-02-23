import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from nilearn import datasets, maskers
from nilearn.surface import vol_to_surf
from nilearn.glm import threshold_stats_img
from nilearn.plotting import plot_surf_stat_map
from nilearn.image import math_img, new_img_like

from fmri_encoder.utils import check_folder

import warnings
warnings.filterwarnings("ignore")

def concat_colormaps(*args, cutting_threshold=75):
    """Concat a list of colormaps.
    Arguments:
        - args: list of str
        - cutting_threshold: int
    Returns:
        - new_cmap: Matplotlib Colormap
    """
    cmaps_tmp = []
    cmaps = []
    for arg in args:
        if isinstance(arg, str):
            cmaps_tmp.append(plt.cm.get_cmap(arg))
        else:
            cmaps_tmp.append(arg)
    for cmap in cmaps_tmp:
        cmaps.append(cmap(np.linspace(0, 1, 255)))
    new_cmap = ListedColormap(np.vstack([cmap[cutting_threshold:] for cmap in cmaps]))
    return new_cmap

def plot_colorbar(cms, data=None, vmax=1, vmin=None):
    """Plot a colormap.
    Arguments: 
        - cms: list of Matplotlib Colormap
    """
    if vmin is None:
        vmin = -vmax
    np.random.seed(19680801)
    if data is None:
        data = np.random.randn(30, 30)

    if type(cms)==list:
        fig, axs = plt.subplots(1, len(cms), figsize=(6, 3), constrained_layout=True)
        for [ax, cmap] in zip(axs, cms):
            psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=vmin, vmax=vmax)
            fig.colorbar(psm, ax=ax)
    else:
        fig, axs = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        psm = axs.pcolormesh(data, cmap=cms, rasterized=True, vmin=vmin, vmax=vmax)
        fig.colorbar(psm, ax=axs)
    plt.show()

def plot_voxels_time_course(fmri_data, saving_folder=None, format_figure='pdf', dpi=100):
    """Plot voxels time-course.
    Args:
        - fmri_data: np.Array, size [#time-points, #voxels]
        - saving_folder: str
    """
    x = input("Are you sure the input has less than 20 voxels ? Otherwise it will crash...\n[yes/no]")
    if x=='yes':
        nb_voxels = fmri_data.shape[-1]
        fig, axs = plt.subplots(nb_voxels, 1, figsize=(25, nb_voxels*4))
        for i in range(nb_voxels):
            axs[i].plot(fmri_data[:, i])
            axs[i].grid(True)
            axs[i].tick_params(axis='both', which='major', labelsize=20)
            axs[i].tick_params(axis='both', which='minor', labelsize=20)
            axs[i].set_ylabel(f'Voxel {i}', fontsize=25)
        axs[nb_voxels-1].set_xlabel('Time', fontsize=25)
        plt.suptitle('Voxels timecourses', fontsize=40)
        if saving_folder is not None:
            check_folder(saving_folder)
            plt.savefig(
                os.path.join(
                    saving_folder, 
                    f'voxels_time_course{format_figure}'
                ), 
                format=format_figure, 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0
            )
        plt.show()
    else:
        print('Reduce it by selecting a subset of voxels.')

def plot_design_matrix(matrix, saving_folder=None, title='Design matrix', format_figure='pdf', dpi=100, figsize=(15, 15)):
    """Plot a  design matrix.
    Args:
        - matrix: np.Array, size [#sample, #features]
        - saving_folder: str
        - title: str
    """
    plt.imshow(matrix)
    plt.xlabel('Features')
    plt.ylabel('Events')
    plt.title(title)
    plt.colorbar()
    if saving_folder is not None:
            check_folder(saving_folder)
            plt.savefig(
                os.path.join(
                    saving_folder, 
                    f'design_matrix{format_figure}'
                ), 
                format=format_figure, 
                dpi=dpi, 
                bbox_inches='tight', 
                pad_inches=0
            )
    plt.show()

def superimpose_imgs_rgb(img1=None, img2=None, img3=None, masker=None):
    """Considering 3 nifti effect-size images, it create a RGB value 
    for each triplet of values and return the associated image and 
    colorbar to be plotted. If less than three images are given, the
    remaining images are considered as full of zeros.
    Args:
        - img1: NiftiImage (Red)
        - img2: NiftiImage (Green)
        - img3: NiftiImage (Blue)
        - masker = NifitMasker
    Returns:
        - img: NiftiImage
        - cmap: Matplotlib ColorMap
    """
    if (img1 is None) and (img2 is None) and (img3 is None):
        raise ValueError("Not images were given as input...")
    ref_img = img1 if img1 is not None else (img2 if img2 is not None else img3)
    if img1 is None:
        img1 = new_img_like(ref_img, np.zeros(ref_img.shape))
    if img2 is None:
        img2 = new_img_like(ref_img, np.zeros(ref_img.shape))
    if img3 is None:
        img3 = new_img_like(ref_img, np.zeros(ref_img.shape))

    if masker is None:
        tmp1 = new_img_like(img1, np.abs(img1.get_fdata()))
        tmp2 = new_img_like(img2, np.abs(img2.get_fdata()))
        tmp3 = new_img_like(img3, np.abs(img3.get_fdata()))
        mask = math_img('img1+img2+img3!=0', img1=tmp1, img2=tmp2, img3=tmp3)
        masker = maskers.NiftiMasker(mask, **{'detrend': False, 'standardize': False, 'standardize_confounds': False})
        masker.fit()
    
    max_value = np.max(np.abs(np.stack([
        masker.transform(img1).reshape(-1),
        masker.transform(img2).reshape(-1), 
        masker.transform(img3).reshape(-1)
    ])))
    
    data = list(zip(
        np.round(masker.transform(img1).reshape(-1)/ max_value, 4), 
        np.round(masker.transform(img2).reshape(-1)/ max_value, 4), 
        np.round(masker.transform(img3).reshape(-1)/ max_value, 4)
        ))
    cmap = [(0., 0., 0., 1)]
    img = []
    print(len(data))
    values = []
    #indexes = np.linspace(-1, 1, len(data))
    indexes = np.arange(1, 1+len(data))
    for i, value in enumerate(data):
        value = (value[0], value[1], value[2], 1)
        if value==(0., 0., 0., 1):
            img.append(0)
        else:
            values.append(value)
            img.append(indexes[i])
        cmap.append(value)
    img = np.array(img) / len(img)
    #cmap = ListedColormap(cmap) # LinearSegmentedColormap.from_list('custom', cmap, N=len(cmap))
    #img = masker.inverse_transform(np.hstack(img))
    #plot_colorbar(cmap)
    return img, cmap, values, masker


def set_projection_params(
    hemi, 
    view, 
    cmap='cold_hot', 
    inflated=False, 
    threshold=1e-15, 
    colorbar=False, 
    symmetric_cbar=False, 
    template=None, 
    figure=None, 
    ax=None, 
    vmax=None
    ):
    """Return the a dict of args to pot surface map (to have a clean function to plot).
    Args:
        - hemi: str
        - view: str
        - cmap: str
        - inflated: bool
        - threshold: float
        - colorbar: bool
        - symmetric_cbar: bool
        - template: 
        - figure: Matplotlib object
        - ax: Matplotlib object
        - vmax: float
    """
    kwargs = {
            'surf_mesh': f'pial_{hemi}', # pial_right, infl_left, infl_right
            'surf_mesh_type': f'pial_{hemi}',
            'hemi': hemi, # right
            'view':view, # medial
            'bg_map': f'sulc_{hemi}', # sulc_right
            'bg_on_data':True,
            'darkness':.7,
                }
    if template is None:
        template = datasets.fetch_surf_fsaverage('fsaverage5')
    if inflated:
        kwargs['surf_mesh'] = 'infl_left' if 'left' in kwargs['surf_mesh_type'] else 'infl_right' 
    surf_mesh = template[kwargs['surf_mesh']]
    bg_map = template[kwargs['bg_map']]

    args = {
        'surf_mesh': surf_mesh, 'hemi': hemi, 
        'view': view, 'bg_map': bg_map, 
        'bg_on_data': kwargs['bg_on_data'], 'darkness': kwargs['darkness'],
        'figure': figure, 'axes': ax, 'vmax': vmax, 'cmap':cmap, 
        'colorbar': colorbar, 'threshold': threshold, 'symmetric_cbar': symmetric_cbar, 
        'bbox_inches': 'tight', 'pad_inches' :  -0.3,
        }
    return args
        
def create_grid(
    nb_rows, 
    nb_columns, 
    zooming_factor=30, 
    row_size_factor=4, 
    overlapping=6, 
    column_size_factor=5, 
    wspace=-0.15, 
    hspace=-0.15, 
    top=1., 
    bottom=0., 
    left=0., 
    right=1
    ):
    """Generate a grid of axes for plotting brain later in.
    Args:
        - nb_rows: int
        - nb_columns: int 
        - zooming_factor: int 
        - row_size_factor: int
        - overlapping: int
        - column_size_factor: int
        - column_size_factor: int
        - wspace: float
        - hspace: float
        - top: float
        - bottom: float
        - left: float
        - right: float
    Returns:
        - Matplotlib.Figure, Matplotlib.Axes objects
    """
    figsize = (column_size_factor*nb_columns, row_size_factor*nb_rows)
    figure = plt.figure(figsize=figsize)
    gs = figure.add_gridspec(
            nb_rows*zooming_factor, 
            (1+nb_columns)*zooming_factor+overlapping, 
            wspace=wspace, hspace=hspace, top=top, bottom=bottom, left=left, right=right)
        
    axes = [
        [
        figure.add_subplot(
            gs[
                int(max(0, i*(zooming_factor-6))):int((i+1)*zooming_factor)-i*6, 
                int(max(0, j*zooming_factor-overlapping)):int((j+1)*zooming_factor+overlapping)
            ], projection= '3d'
            ) for j in range(1, 1+nb_columns)
        ] for i in range(nb_rows)
    ]
    # Setting background to transparent
    for ax in axes:
        for i in ax:
            i.patch.set_alpha(0)
    return figure, axes

def compute_surf_proj(
    imgs, 
    zmaps, 
    masks, 
    names, 
    ref_img, 
    categorical_values=None, 
    inflated=False, 
    hemispheres=['left', 'right'], 
    views=['lateral', 'medial'], 
    kind='line', 
    template=None):
    """ Compute all surface projection for the images, views and hemispheres given.
    Args:
        - imgs:
        - masks:
        - names: 
        - hemispheres: list of str
        - categorical_values: list of bool
        - kind: str
        - views: list of str
        - inflated: bool
        - template:
        - *kwargs: 
    Returns:
        - surf_img: list of np.arrray (input image projected onto the surface)
    """
    result = {}
    categorical_values = [False]*len(imgs) if categorical_values is None else categorical_values
    masks = [None]*len(imgs) if masks is None else masks
    for index, (img, mask) in enumerate(zip(imgs, masks)):
        ref_img = math_img('img!=0', img=img) if ref_img is None else ref_img
        result[names[index]] = {}
        for h, hemi in enumerate(hemispheres):
            for v, view in enumerate(views):
                kwargs = {
                    'surf_mesh': f'pial_{hemi}', # pial_right, infl_left, infl_right
                    'surf_mesh_type': f'pial_{hemi}',
                    'hemi': hemi, # right
                    'view':view, # medial
                    'bg_map': f'sulc_{hemi}', # sulc_right
                    'bg_on_data':True,
                    'darkness':.7,
                }
                if zmaps is not None:
                    thresholded_zmap, fdr_th = threshold_stats_img(
                                                stat_img=math_img('img1*img2', img1=zmaps[index], img2=ref_img), 
                                                alpha=0.1, 
                                                height_control='bonferroni',
                                                cluster_threshold=30,
                                                mask_img=ref_img)
                    mask_bonf = new_img_like(img, (np.abs(thresholded_zmap.get_fdata()) > 0))
                    mask = math_img('img1*img2', img1=mask_bonf, img2=ref_img)
                result[names[index]][f'{hemi}-{view}'] = proj_surf(
                    img, 
                    mask=mask, 
                    template=template, 
                    inflated=inflated, 
                    categorical_values=categorical_values[index], **kwargs)
    return result

def proj_surf(img, mask=None, template=None, kind='line', inflated=False, **kwargs):
    """Project a volumic image onto brain surface.
    Args:
        - img:
        - mask:
        - template:
        - kind: str
        - inflated: bool
        - *kwargs: 
    Returns:
        - surf_img: np.arrray (input image projected onto the surface)
    """
    if template is None:
        template = datasets.fetch_surf_fsaverage('fsaverage5')
    if inflated:
        kwargs['surf_mesh'] = 'infl_left' if 'left' in kwargs['surf_mesh_type'] else 'infl_right' 
    surf_mesh = template[kwargs['surf_mesh']]
    #bg_map = template[kwargs['bg_map']]

    surf_img = vol_to_surf(img, surf_mesh, mask_img=mask, interpolation='nearest', kind=kind, radius=1e-15, n_samples=10)
    surf_img[surf_img==0] = np.nan

    return surf_img


def pretty_plot(
    imgs, 
    zmaps, 
    masks,
    names,
    ref_img=None,
    vmax=[0.2], 
    cmap='cold_hot',
    hemispheres=['left', 'right'], 
    views=['lateral', 'medial'], 
    categorical_values=None, 
    inflated=False, 
    saving_folder='./derivatives/', 
    format_figure='pdf', 
    dpi=300, 
    plot_name='test',
    symmetric_cbar=False,
    colorbar=False,
    row_size_factor=4,
    overlapping=6,
    column_size_factor=5,
    ):
    """
    """
    surf_imgs = compute_surf_proj(
        imgs, 
        zmaps=zmaps, 
        masks=masks, 
        ref_img=ref_img, 
        names=names, 
        categorical_values=categorical_values, 
        inflated=inflated,
        hemispheres=hemispheres, 
        views=views, 
        kind='line', 
        template=None
        )


    figure, axes = create_grid(
        nb_rows=len(names), 
        nb_columns=4, 
        row_size_factor=row_size_factor, 
        overlapping=overlapping, 
        column_size_factor=column_size_factor
        )
    positions = {
        'lateral-left': 0,
        'medial-left': 1,
        'medial-right': 2,
        'lateral-right': 3,
    }

    for i, name in enumerate(names):
        for h, hemi in enumerate(hemispheres):
            for v, view in enumerate(views):
                ax = axes[i][positions[f"{view}-{hemi}"]]
                kwargs = set_projection_params(hemi, view, cmap=cmap, 
                inflated=inflated, threshold=1e-15, colorbar=colorbar, symmetric_cbar=symmetric_cbar, template=None, figure=figure, ax=ax, vmax=vmax[i])

                surf_img = surf_imgs[name][f'{hemi}-{view}']
                plot_surf_stat_map(stat_map=surf_img,**kwargs)
    
    check_folder(os.path.join(saving_folder, 'figures'))        
    plt.savefig(os.path.join(saving_folder, 'figures', f'{plot_name}.{format_figure}'), format=format_figure, dpi=dpi, bbox_inches = 'tight', pad_inches = 0, )
    plt.show()
    plt.close('all')
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['backend'] 
from matplotlib.widgets import Slider
import numpy as np
from scipy.interpolate import UnivariateSpline
import holoviews as hv
import panel as pn
import plotly
import plotly.offline as offline
import mpld3
import myServices as ms
import models as m
from collections import OrderedDict

from bokeh.palettes import Spectral11
from bokeh.layouts import row,column
import bokeh.plotting as bk
from bokeh.resources import INLINE
from bokeh.plotting import figure, output_file, save, show, gridplot
from bokeh.models import Slider,ColumnDataSource, CustomJS, HoverTool, LinearColorMapper,ColorBar, BasicTicker
from bokeh.transform import factor_cmap, linear_cmap
bk.output_notebook() #for viz within ipython notebook

# # Transformation 
# def probs_to_labels(pos_probs, threshold):
#     return (pos_probs >= threshold).astype(float)

# image_array = np.random.uniform(low=-100, high=100, size=(100,100))

# # Crear un subplot con 1 fila y 1 columna
# fig, ax = plt.subplots(1, 1)
# # Mostrar la imagen en el subplot
# ax.set_title("Flood Susceptibility")
# ax.set_xticks([])  # Ocultar etiquetas del eje x
# ax.set_yticks([])  # Ocultar etiquetas del eje y

# # Añadir una barra de color
# ## Add control bar
# plt.subplots_adjust(bottom=0.25)
# ax_slider = plt.axes([0.25,0.1,0.5,0.03]) ## x_Position,y_position,wigth, hight
# s_factor = Slider(ax_slider, 'SigmoidThreshold', valmin=0.1,valmax=1, valinit=0.5, valstep=0.02)
# imageToPlot = probs_to_labels(image_array,s_factor.val)
# ax.imshow(imageToPlot, cmap='gray')
# del imageToPlot

# def updateImge(val):
#     imageToPlot = probs_to_labels(image_array,s_factor.val)
#     ax.imshow(imageToPlot,cmap='gray')
#     del imageToPlot
#     fig.canvas.draw()  ## Redraw the figure

# s_factor.on_changed(updateImge)

# plt.show()
############### __________________________________!!!!!!!!!!!  Testing Zone
# Definir función para actualizar el gráfico según el valor del slider


# Generar un array nxn con valores aleatorios entre 0 y 1

def genrateInteractivePlot(maskTifPath,probTifPath,outputPath):

    ## Import rasters as array
    mask_array = ms.readRasterAsArry(maskTifPath)
    mask_array = (np.where(mask_array<0,0,mask_array)).astype(int)
    # print(f'mask shape {mask_array.shape} and type {mask_array.dtype}')
    # print(np.unique(mask_array))
    predictedProba = ms.readRasterAsArry(probTifPath)
    shape = predictedProba.shape
    # print(f'array shape {shape} and type {predictedProba.dtype}')

    # Create initial images 
    output_file(outputPath)
    fig = figure()
    mask = ColumnDataSource(data={'mask': [mask_array]})
    # Crear la fuente de datos para la image

    tresholdedProba = (predictedProba>=0.5).astype(int)
    source = ColumnDataSource(data={'image': [tresholdedProba], 'initial_image': [predictedProba]})

    ## Plot ground truth mask. 
    fig.image(image='mask', source=mask, x=0, y=0, dw=shape[0], dh=shape[1], palette=["white",'#000000'])#,legend_label="Mask"

    ## Uncomment if desired  # Plot original prediction as a permanent reference.
    # fig.image(image='initial_image', source=source, x=0, y=0, dw=shape[0], dh=shape[1], palette=["white", '#000000'])

    ## Plot dinamic prediction. 
    fig.image(image='image', source=source, x=0, y=0, dw=shape[0], dh=shape[1], palette=["white", "red"], alpha=0.5,legend_label="Interactive prediction")

    # Definir la barra deslizante (slider)
    slider = Slider(start=0, end=1, value=0.5, step=0.01, title="Threshold")

    # Create confusion matrix
    maskFlat = mask_array.flatten()#mask.data['mask']#
    probaFlat = tresholdedProba.flatten()#source.data['image']#
    mat = m.confusion_matrix(maskFlat,probaFlat)
    print(f"Original Conf Matrix : {mat}")
    
    confMat = createConfMatrixFig(mat) # plotInteractiveConfusionMatrix(mat)

    # Función para actualizar la imagen según el valor del slider
    update_image_js = CustomJS(args=dict(source=source, slider=slider, mask=mask), code="""
        const data = source.data['image'];
        const initial_data = source.data['initial_image'];
        const value = slider.value;
        for (let i = 0; i < data.length; i++) {
            for (let j = 0; j < data[i].length; j++) {
                if (initial_data[i][j] >= value) {
                    data[i][j] = 1;
                } else {
                    data[i][j] = 0;
                }
            }
        }
        source.change.emit();                         
    """)

    slider.js_on_change('value', update_image_js)

    ### Update Confusion Matrix:
    # Define custom JavaScript callback function
    callback = CustomJS(args=dict(source=source, mask=mask),code="""
        function flattenArray(array) {
            return array.reduce((acc, val) => acc.concat(val), []);
        }
                                                   
        function calculateConfusionMatrix(y_real, y_hat) {
            const trueVals = flattenArray(y_real);
            const predicted = flattenArray(y_hat);
                                                
            if (trueVals.length !== predicted.length) {
                throw new Error("Vectors lengh do not match");
            }
                               
            const tPosit = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 1 && predicted[idx] === 1 ? count + 1 : count;
            }, 0);
            const fPosit = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 0 && predicted[predicted] === 1 ? count + 1 : count;
            }, 0);

            const tNeg = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 0 && predicted[predicted] === 0 ? count + 1 : count;
            }, 0);

            const fNeg = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 1 && predicted[predicted] === 0 ? count + 1 : count;
            }, 0);

            return {
                TP: tPosit,
                FP: fPosit,
                TN: tNeg,
                FN: fNeg
            };
        }
        
        const data = source.data['image'];
        const mask = mask.data['mask']; 
        TN,FN,TP,FP = calculateConfusionMatrix( data, mask);
        mask[0,0] = TN;
        mask[0,1] = FN;
        mask[1,0] = TP;
        mask[1,1] = FP;
        console.log(TN,FN,TP,FP);
        confMat.change.emit();
                        
        """
    )
    # Attach the JavaScript callback to the ColumnDataSource
    source.js_on_change('data', callback)



    # Crear el layout
    
    layout = row(children=[column(fig,slider), confMat])# confMatrix,

    ## Setting leyend
    # Personalizar la ubicación de la leyenda
    fig.legend.location = "top_left"
    # fig.legend.title = "Observaciones"

    # Cambiar la apariencia del texto de la leyenda
    fig.legend.label_text_font = "times"
    fig.legend.label_text_font_style = "italic"
    fig.legend.label_text_color = "navy"

    # Cambiar el borde y el fondo de la leyenda
    # fig.legend.border_line_width = 3
    # fig.legend.border_line_color = "navy"
    # fig.legend.border_line_alpha = 0.8
    # fig.legend.background_fill_color = "navy"
    fig.legend.background_fill_alpha = 1
    
    show(layout,sizing_mode="scale_width")
    # Guardar la figura HTML
    # save(layout)
    
 
def createConfMatrixFig(matrix):
    # Create the Bokeh figure
    p = figure(title='Confusion Matrix', x_axis_label='Predicted Labels', y_axis_label='True Labels',x_range=(-0.5, 1.5), y_range=(-0.5, 1.5))
    p.width = 400
    p.height = 400

    # Create the color mapper with a predefined palette
    palette = ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#eff3ff']
    mapper = linear_cmap(field_name='value', palette=palette, low=1, high=0)

    # Flatten the matrix and create ColumnDataSource
    percentMAtrix =  matrix/np.sum(matrix)
    # print(f"Percentage Conf Matrix : {percentMAtrix}")
    flat_matrix_Percent = [percentMAtrix[0,0],percentMAtrix[0,1],percentMAtrix[1,0],percentMAtrix[1,1]]
    #[0.9507904128635555, 0.036010381519726235, 0.011109304862189034, 0.0020899007545292733]
    # print(f"flat_matrix_Percent : {flat_matrix_Percent}")
    
    actual= [0, 0, 1, 1]
    predicted= [0, 1, 0, 1]
    labels = ['True Neg', 'False Pos', 'False Neg','True Pos',]
    counts = [f"{value:.3f}" for value in flat_matrix_Percent]
    percentages = [f"{l}\n {(v ):.2%}" for l,v in zip(labels,flat_matrix_Percent)]
    source = ColumnDataSource(data=dict(predicted=predicted, actual=actual, labels=labels, counts=counts,percentages=percentages, value=flat_matrix_Percent))

    # Add rectangles to the plot
    r = p.rect(x='predicted', y='actual', width=1, height=1, source=source,
            fill_color=mapper, line_width=1, line_color='black')

    # Add text annotations
    p.text(x='predicted', y='actual', text= 'percentages', source=source,
            text_align='center', text_baseline='middle', text_color='black')

    # Add color bar
    # color_bar = ColorBar(color_mapper=mapper['transform'], width=6, location=(0, 0),
                            # ticker=BasicTicker(desired_num_ticks=4))
    color_bar = r.construct_color_bar(width=10)  ## Equivalent to 2 previous lines
    
    p.add_layout(color_bar, 'right')


    # Customize plot
    p.xaxis.ticker = [0, 1]
    p.xaxis.major_label_overrides = {0: '0', 1: '1'}
    p.yaxis.ticker = [0, 1]
    p.yaxis.major_label_overrides = {0: '0', 1: '1'}
    
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 1
    # p.xaxis.major_label_orientation = np.pi / 3

    return p 

def monitor_data_changes(source, mask):
    """
    Monitor changes in the data of ColumnDataSource and recalculate confusion matrix.

    Parameters:
    source (ColumnDataSource): The ColumnDataSource to monitor.Must contain :
        PREDICTED -> = source.data['image'];

    mask (ColumnDataSource): The ColumnDataSource to get the mask from :
        real-> = mask.data['mask'];
    Returns:
    None
    """
    # Define custom JavaScript callback function
    callback = CustomJS(args=dict(source=source, mask=mask),code="""
        function flattenArray(array) {
            return array.reduce((acc, val) => acc.concat(val), []);
        }
                                                   
        function calculateConfusionMatrix(y_real, y_hat) {
            const trueVals = flattenArray(y_real);
            const predicted = flattenArray(y_hat);
                                                
            if (trueVals.length !== predicted.length) {
                throw new Error("Vectors lengh do not match");
            }
                               
            const tPosit = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 1 && predicted[idx] === 1 ? count + 1 : count;
            }, 0);
            const fPosit = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 0 && predicted[predicted] === 1 ? count + 1 : count;
            }, 0);

            const tNeg = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 0 && predicted[predicted] === 0 ? count + 1 : count;
            }, 0);

            const fNeg = trueVals.reduce((count, trueVals, predicted) => {
                return trueVals === 1 && predicted[predicted] === 0 ? count + 1 : count;
            }, 0);

            return {
                TP: tPosit,
                FP: fPosit,
                TN: tNeg,
                FN: fNeg
            };
        }
        
        const data = source.data['image'];
        const mask = mask.data['mask']; 
        TN,FN,TP,FP = calculateConfusionMatrix( data, mask);
        mask[0,0] = TN;
        mask[0,1] = FN;
        mask[1,0] = TP;
        mask[1,1] = FP;
        console.log(TN,FN,TP,FP);
        confMat.change.emit();
                        
        """
    )
    # Attach the JavaScript callback to the ColumnDataSource
    source.js_on_change('data', callback)


localTif =r'C:\Users\abfernan\CrossCanFloodMapping\GitLabRepos\FloodProbabRNCanAbd\local\temp.tif'
local_mask = r'C:\Users\abfernan\CrossCanFloodMapping\GitLabRepos\FloodProbabRNCanAbd\local\tempMask_16m.tif'
local_html_Image =r'C:\Users\abfernan\CrossCanFloodMapping\GitLabRepos\FloodProbabRNCanAbd\local\tempPlusMask_16m_With_CM.html'
genrateInteractivePlot(local_mask,localTif, local_html_Image)

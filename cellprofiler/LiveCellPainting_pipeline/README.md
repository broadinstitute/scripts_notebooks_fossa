# Analysis pipeline example: Live Cell Painting

analysis.cppipe contains an example pipeline to be used on CellProfiler. 

## Requirements

1. A load_csv.csv file that contains the Metadata_Plate, Metadata_Well, Metadata_Site, FileName and PathName for each channel. To generate this file, you can use the notebook available at https://github.com/fefossa/LoadDataGenerator.git 

2. Images of cells stained with Acridine Orange, because the model for nuclei detection with Cellpose and identification of acidic vesicles and nucleoli are based on those types of images.

3. Install CellProfiler from source to use RunCellpose for nuclei detection. For that, follow the instructions on https://github.com/CellProfiler/CellProfiler-plugins.
import easygui as eg
import os
import openpyxl
from metadata_utils import plate_wells, get_samples, get_example_to_name_metadata_cols, generate_rows_lists, generate_csv

def main():
    print("### Metadata From Layout takes an Excel file with a plate layout and creates a CSV/txt file with your metadata in it ###")
    print("\n **INSTRUCTION**: Enter the number of wells in your plate \n")
    
    well_plate_input = eg.enterbox("Enter the well-plate number (6, 12, 24 or 96)")
    well_plate = int(well_plate_input)

    wells = plate_wells(plate_type=well_plate)

    pathname = eg.fileopenbox("**INSTRUCTION**: Choose layout file (from Excel) \n")
    file_name, extension = os.path.splitext(os.path.basename(pathname))
    print(f"\n INPUT FOLDER: {pathname} \n")

    reading = openpyxl.load_workbook(pathname)
    layout = reading.active 
    samples = get_samples(layout, plate_type=well_plate)

    print("\n **INSTRUCTION**: Write the name of each column corrresponding to the example sample \n")
    metadata_cols = get_example_to_name_metadata_cols(samples)

    sublist = generate_rows_lists(samples, wells)

    # add platemap_name
    metadata_cols.insert(0, "plate_map_name")
    sublist_add_platemap = [[file_name] + sublist for sublist in sublist]

    # finally, generate csv and txt files
    generate_csv(file_name, metadata_cols, sublist_add_platemap)

    print("\n ## Finished successfully. ## \n ## Check your output folder/platemap! ##")

if __name__ == "__main__":
    main()
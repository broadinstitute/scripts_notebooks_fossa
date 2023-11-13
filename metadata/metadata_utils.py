def plate_wells(plate_type=96):
    """ 
    Provide a plate type and return the well's names
    plate_type accepted (int): 96, 24, 12, 6
    """
    import string
    import itertools

    if plate_type == 24:
        number_cols = 6   # The ending number in the range (inclusive)
        number_rows = 4
    elif plate_type == 96:
        number_cols = 12   # The ending number in the range (inclusive)
        number_rows = 8
    elif plate_type == 12:
        number_cols = 4   # The ending number in the range (inclusive)
        number_rows = 3
    elif plate_type == 6:
        number_cols = 3   # The ending number in the range (inclusive)
        number_rows = 2
    
    columns = [str(num) for num in range(1, number_cols + 1)]
    rows = list(string.ascii_uppercase[:number_rows])

    wells = []
    for pair in itertools.product(rows, columns):
        wells.append(''.join(pair))
    
    print(f"Number of wells: {len(wells)}")
    print(wells)
    
    return wells

def get_samples(layout, plate_type=96):
    """ 
    """
    if plate_type == 24:
        number_cols = 6   # The ending number in the range (inclusive)
        number_rows = 4
    elif plate_type == 96:
        number_cols = 12   # The ending number in the range (inclusive)
        number_rows = 8
    elif plate_type == 12:
        number_cols = 4   # The ending number in the range (inclusive)
        number_rows = 3
    elif plate_type == 6:
        number_cols = 3   # The ending number in the range (inclusive)
        number_rows = 2
    names = []
    for row in layout.iter_rows(max_row=number_rows,max_col=number_cols): #iterate through rows until 8 and cols till 12 (size of the 96 well plate)
        for cell in row: #iterate within the columns in each row
            names.append(str(cell.internal_value)) #append the cell values to the names list
    names_return = [t.rstrip() for t in names] #remove any spaces at the beggining or end of the string
    print(names_return)
    return names_return

def generate_rows_lists(samples, wells):
    """
    """

    sublist = []
    for s in range(len(samples)):
        if s != 'None':
            itens = samples[s].split()
            itens.insert(0, wells[s])
            sublist.append(itens)
        else:
            sublist.append([])
    return sublist
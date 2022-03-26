from typing import List, Tuple, Union, TextIO
### You may import any Python's standard library here (Do not import other external libraries) ###

### Import End ###

class Baskets:
    def __init__(self, filePath: str) -> None:
        self.fd: TextIO = open(filePath, "r")  # File object
        ############## TODO: Complete the function (Optional) ##################
        # You may declare any class variables if needed                        #
        ########################################################################

        ######################### Implementation end ###########################

    def __del__(self) -> None:
        self.fd.close()

    # Read one basket from the file
    def readItems(self) -> Union[Tuple[int, ...], None]:
        #################### TODO: Complete the function #######################
        # Read one line from the file object                                   #
        # If you cannot read it, then return None                              #
        # Otherwise, parse the line then return the tuple (itemid, ...)        #
        itemID = list()
        line = self.fd.readline()
        if line:
            items = line.strip().split(' ')
            for item in items:
                itemID.append(int(item))
            return tuple(sorted(itemID))
        ########################################################################

        return None

        ######################### Implementation end ###########################

    # Move the file pointer into the beginning of the file
    def setBOF(self) -> None:
        self.fd.seek(0)
        ############## TODO: Complete the function (Optional) ##################
        # You may modify the function if needed                                #
        ########################################################################

        ######################### Implementation end ###########################

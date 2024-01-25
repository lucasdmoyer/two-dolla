from backtrader.feeds import GenericCSVData

class GenericCSV_Position(GenericCSVData):

    # Add a 'pe' line to the inherited ones from the base class
    lines = ('position',)

    # openinterest in GenericCSVData has index 7 ... add 1
    # add the parameter to the parameters inherited from the base class
    params = (('position', 8),)
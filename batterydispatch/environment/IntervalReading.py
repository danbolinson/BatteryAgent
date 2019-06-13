from requests import get
import xml.etree.ElementTree
import pandas as pd
import matplotlib.pyplot as plt

class IntervalReading:
    '''Loads in GreenButton interval data from an xml file in the given location, and makes full load or monthly
    load data available as a dataframe.'''

    def __init__(self, green_button_xml_file=""):
        '''The xml file provided is read and the initialize function is called to parse the data into a dataframe.'''
        self.duration = 0
        self.start = 0
        self.readings = {}
        self.DF = pd.DataFrame()
        if green_button_xml_file is not None:
            self.xml = xml.etree.ElementTree.ElementTree(file=green_button_xml_file)
            self.initialize(self.xml)

    def initialize(self, etree):
        '''Converts the xml file into a dataframe and formats the values and dates appropriately.'''
        root = etree.getroot()
        iblock = root[7][2][0]
        for i in iblock[0]:
            if 'duration' in i.tag:
                self.duration = i.text
            elif 'start' in i.tag:
                self.start = i.text

        namespace = iblock.tag[0:iblock.tag.find('}') + 1]
        readings = {}
        for ix, block in enumerate(iblock.findall(namespace + 'IntervalReading')):
            readings[ix] = {}
            for data in block:
                if 'timePeriod' in data.tag:
                    for t in data:
                        if 'duration' in t.tag:
                            readings[ix]['duration'] = t.text
                        elif 'start' in t.tag:
                            readings[ix]['start'] = t.text
                elif 'value' in data.tag:
                    readings[ix]['value'] = data.text

        self.readings = readings
        self.DF = self.DF.from_dict(readings, orient='index')
        self.DF.start = pd.to_datetime(self.DF.start, unit='s')
        self.DF.value = self.DF.value.astype(float)
        self.DF['duration_hrs'] = (self.DF.start - self.DF.start.shift()).fillna(
            pd.Timedelta(seconds=0)).dt.total_seconds() / 3600

    def plot_load(self, start=0, end=-1):

        plt.plot(self.readings.DF.value[start:end])
        plt.show()

    def get_month_generator(self):
        '''Returns a generator function which yields a DataFrame for a 30 days of load'''

        def split_into_30days(series):
            start = series[0].date()
            end = start + pd.Timedelta(30, unit='D')
            chunks = series.copy()
            for i, d in enumerate(series):
                if d.date() > end:
                    start = end + pd.Timedelta(1, unit='D')
                    end = start + pd.Timedelta(30, unit='D')

                chunks[i] = start
            return chunks

        self.DF['billing_period'] = split_into_30days(self.DF.start)

        for bill in list(set(self.DF.billing_period)):
            yield (self.DF[self.DF.billing_period == bill])

        return

    def get_daily_generator(self):
        '''Returns a generator function which yields a DataFrame for 1 day of load.
        Note there is no guarantee that the day is in the same month.'''

        self.DF['start_date'] = [s.date() for s in self.DF.start]

        for day in list(set(self.DF.start_date)):
            yield (self.DF[self.DF.start_date == day])

        return


    def get_max_load(self):
        '''Returns the MAXIMUM value over the entire period (NOT any specific month.)'''
        return max(self.DF['value'])

    def get_min_load(self):
        '''Returns the MINIMUM value over the entire period (NOT any specific month.)'''
        return min(self.DF['value'])


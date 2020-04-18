"""
Module for functions used in analysis of Chernobyl air data.
"""

# General math
import numpy as np
import math

# Map operations
from shapely.geometry import Point
import geopandas as gpd

# TODO write test suites for each function

def read_world_map(shapefile):
    """ Read a shapefile to a geo data frame.

    :param shapefile: A shapefile containing world map.
    :return: A geo data frame containing world data.
    """

    gdf_world = gpd.read_file(shapefile)[['ADMIN', 'ISO_A2', 'geometry']]
    gdf_world.columns = ['country', 'country_code', 'geometry']
    return gdf_world

def check_countries(dataFrameOne, dataFrameTwo):
    """ Check that both data frames have the same countries based on country code.
    Note that smaller data frame should go first.

    :param dataFrameOne: Smaller data frame.
    :param dataFrameTwo: Bigger data frame.
    :return: Boolean based on whether country codes in first df are in second.
    """
    countries_one = set(dataFrameOne.country_code)
    countries_two = set(dataFrameTwo.country_code)
    set_difference = countries_one.difference(countries_two)
    if set_difference == set():
        return True
    print(set_difference)
    return False

    # Test one: insure a country code exists for the frames

def set_france_uk(data_frame):
    """ Change country codes for France and UK in Chernobyl air data.

    :param data_frame: Chernobyl air data frame.
    :return: Chernobyl air data frame containing FR for France and GB for United Kingdom.
    """
    data_frame.loc[data_frame['country_code'] == 'F', 'country_code'] = 'FR'
    data_frame.loc[data_frame['country_code'] == 'UK', 'country_code'] = 'GB'
    return data_frame

def set_norway_france(gdf_world):
    """ The world map provided by natural earth at a 50 mm resolution has -99 entries for Norway and
    France. This function sets them straight.

    :param gdf_world: Geo data frame for the whole world.
    :return: Geo data frame with Norway and France marked in country codes.
    """
    gdf_world.loc[gdf_world['country'] == 'France', 'country_code'] = 'FR'
    gdf_world.loc[gdf_world['country'] == 'Norway', 'country_code'] = 'NO'
    return gdf_world

    # Test suite

def make_europe_map(df_chern, gdf_world):
    """ Cut the world map to only find countries affected by Chernobyl.

    :param df_chern: Chernobyl air data frame.
    :param gdf_world: Geo data frame for the whole world.
    :return: A geo data frame containing countries affected by Chernobyl air data.
    """

    chern_countries = set(df_chern.country_code)

    europe_indices = []
    for index in range(len(gdf_world)):
        if gdf_world.country_code.values[index] in chern_countries:
            europe_indices.append(index)
    gdf_europe = gdf_world.iloc[europe_indices]

    # Chernobyl accident happened in Ukraine
    gdf_europe = gdf_europe.append(gdf_world.loc[gdf_world.country == 'Ukraine'])

    gdf_europe.index = range(len(gdf_europe))

    return gdf_europe

def make_gdf_air(df_chern):
    """ Turn the chernobyl concentration data into a geo pandas data frame.

    :param df_chern: Data frame containing Chernobyl air concentration data.
    :return: A geo data frame of the Chernobyl data.
    """

    gdf_air = gpd.GeoDataFrame(df_chern)
    crs = 'epsg:4326'
    geometry = [Point(xy) for xy in zip(df_chern["latitude"], df_chern["longitude"])]
    gdf_air = gpd.GeoDataFrame(df_chern, crs=crs, geometry=geometry)
    return gdf_air

def find_low_points(gdf):
    """ Takes a geo data frame and finds all points whose latitude coordinate is less than 22 degrees.

    :param gdf: Geo data frame with lat measurements.
    :return: List of indices whose coordinates lie below 22 degrees lat.
    """
    index_counter = 0
    index_list = []
    for point in gdf.geometry.values:
        if point.xy[1][0] < 22:    # Access the latitude coordinate of a point
            index_list.append(index_counter)
        index_counter += 1
    return index_list

def modify_vaernes_valencia(gdf):
    """ Resets latitude and longitude coordinates for Vaernes and Valencia data. This includes a modification of the geometric column.

    :param gdf: Geo data frame containing data for Vaernes, NO and Valencia, ES.
    :return: Geo data frame with correct measures for Vaernes and Valencia.
    """

    gdf.loc[gdf.city == 'VAERNES', 'latitude'] = 10.327740
    gdf.loc[gdf.city == 'VAERNES', 'longitude'] = 59.199860

    gdf.loc[gdf.city == 'VALENCIA', 'latitude'] = -0.376288
    gdf.loc[gdf.city == 'VALENCIA', 'longitude'] = 39.469906

    # Must modify since otherwise plotting would not work
    geometry = [Point(xy) for xy in zip(gdf["latitude"], gdf["longitude"])]
    gdf.geometry = geometry

    return gdf

def find_out_indices_cz(gdf):
    """ Finds indices for entries of CZ that are not correct.

    :param gdf: Geo data frame containing an entry for CZ.
    :return: A list of indices where lat and long's are outside of the range for CZ.
    """

    # Ranges for CZ only.
    lat_range = [12, 19]
    long_range = [48.5, 51]

    gdf_cz = gdf.loc[gdf.country_code == 'CZ']
    out_indices = []
    for index in gdf_cz.index:
        current_lat = gdf.iloc[index].latitude
        current_long = gdf.iloc[index].longitude
        if current_lat < lat_range[0] or current_lat > lat_range[1]:            # Falls out of range of latitude
            if current_long < long_range[0] or current_long > long_range[1]:    # Falls out of range of longitude
                out_indices.append(index)
    return out_indices

def swap_lat_long(gdf, indices):
    """ Swap the lat and long of a data frame.

    :param gdf: A geo data frame with latitude and longitude.
    :param indices: Indices at which to swap lat and long.
    :return: Geo data frame with swapped lat and long.
    """

    gdf['temp'] = gdf.loc[indices, 'latitude']
    gdf.loc[indices, 'latitude'] = gdf.loc[indices, 'longitude']
    gdf.loc[indices, 'longitude'] = gdf.loc[indices, 'temp']
    gdf.drop('temp', axis=1, inplace=True)

    geometry = [Point(xy) for xy in zip(gdf["latitude"], gdf["longitude"])]
    gdf.geometry = geometry

    return gdf

def change_au(gdf_europe, gdf_world):
    """ Replace Australia's entry with Austria in european geo data frame.

    :param gdf_europe: Geo data frame containing european countries affected by Chernobyl.
    :param gdf_world: Geo data frame for the whole world.
    :return: Geo data frame with Austria instead of Australia.
    """

    au_index = gdf_europe.index[gdf_europe.country == 'Australia']
    au_index = au_index[0]  # Access the actual index value

    gdf_europe.at[au_index, 'country'] = 'Austria'
    gdf_europe.at[au_index, 'country_code'] = 'AT'
    gdf_europe.at[au_index, 'geometry'] = gdf_world.loc[gdf_world.country_code == 'AT', 'geometry'].values[0]

    return gdf_europe

def change_ir(gdf_europe, gdf_world):
    """ Remove Iran from european geo data frame. Add in entry for Ireland.

    :param gdf_europe: Geo data frame containing european countries affected by Chernobyl.
    :param gdf_world: Geo data frame for the whole world.
    :return: Geo data frame with entry for Ireland. (Removed Iran).
    """
    ir_index = gdf_europe.index[gdf_europe.country_code == 'IR']
    ir_index = ir_index[0]

    gdf_europe = gdf_europe.drop(ir_index)
    gdf_europe = gdf_europe.append(gdf_world.loc[gdf_world.country == 'Ireland'])
    gdf_europe.index = range(len(gdf_europe))

    return gdf_europe

def change_slovakia(gdf_air, gdf_world):
    """ Modify the CHernobyl air geo data frame to have Slovakia.

    :param gdf_air: Geo data frame containing Chernobyl air concentrations.
    :param gdf_world: Geo data frame of the whole world.
    :return: Modified geo data frame with Slovakian country code.
    """
    slovakia_data = gdf_world.loc[gdf_world.country == 'Slovakia']

    for row_num in range(len(gdf_air)):
        point_interest = gdf_air.iloc[row_num].geometry
        if slovakia_data.geometry.contains(point_interest).values[0]:    # Does Slovakia contain the lat long of point?
            gdf_air.iat[row_num, 1] = 'SK'    # 1 to access country code

    return gdf_air

def add_slovakia(gdf_europe, gdf_world):
    """ Add Slovakia to european geo data frame.

    :param gdf_europe: Geo data frame containing european countries affected by Chernobyl.
    :param gdf_world: Geo data frame for the whole world.
    :return: Geo data frame with Slovakia included.
    """

    slovakia_data = gdf_world.loc[gdf_world.country == 'Slovakia']
    gdf_europe = gdf_europe.append(slovakia_data)
    gdf_europe.index = range(len(gdf_europe))

    return gdf_europe

def include_lux(gdf_europe, gdf_world):
    """ Add Luxembourg to european geo data frame.

    :param gdf_europe: Geo data frame containing european countries affected by Chernobyl.
    :param gdf_world: Geo data frame for the whole world.
    :return: Geo data frame with Luxembourg included.
    """
    lux_data = gdf_world.loc[gdf_world.country_code == 'LU']
    gdf_europe = gdf_europe.append(lux_data)
    gdf_europe.index = range(len(gdf_europe))
    return gdf_europe

def fill_a_point(index, df, concentration_type):
    """ Create a point containing Chernobyl air concentration data.

    :param index: Numeric index in the data frame.
    :param df: The Chernobyl air data set as a data frame.
    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :return: A dictionary containing values for a single measurement.
    """
    coordinates = [df.loc[index, 'latitude'], df.loc[index, 'longitude']]

    if concentration_type == 'i131':
        concentration = df.loc[index, 'i131']
    elif concentration_type == 'cs134':
        concentration = df.loc[index, 'cs134']
    else:
        concentration = df.loc[index, 'cs137']

    concentration_paragraph, date_paragraph, header = popup_html(concentration, df, index)

    popup = header + concentration_paragraph + date_paragraph

    time = df.loc[index, 'Date']

    result = {'coordinates': coordinates,
              'popup': popup,
              'time': time,
              'concentration': concentration,
              'concentration_type': concentration_type}
    return result

def popup_html(concentration, df, index):
    """ Create the string that goes in the popup of a point.

    :param concentration: Numerical air concentration value.
    :param df: Chernobyl air data set as a data frame.
    :param index: Numerical index in the data frame.
    :return: String containing HTML data for a point popup.
    """
    begin_head_wrap = '<h3 style="font-family:Times New Roman;"> '
    city = df.loc[index, 'city'].lower()
    city = city.capitalize()
    country_code = df.loc[index, 'country_code']
    end_head_wrap = ' </h3>'
    header = begin_head_wrap + city + ', ' + country_code + end_head_wrap

    conc_begin_par_wrap = '<p style="font-family:Times New Roman;">'
    rounded_value = np.round(concentration, 4)
    str_concentration_value = str(rounded_value)
    conc_end_par_wrap = '</p>'
    concentration_paragraph = conc_begin_par_wrap + 'Concentration: ' + str_concentration_value + conc_end_par_wrap

    date_begin_par_wrap = '<p style="font-family:Times New Roman;">'
    str_date = str(df.loc[index, 'Date'])
    date_end_par_wrap = '</p>'
    date_paragraph = date_begin_par_wrap + 'Date: ' + str_date + date_end_par_wrap

    return concentration_paragraph, date_paragraph, header

def determine_color(concentration, concentration_type):
    """ Return a color based on the intensity of the concentration.

    :param concentration: Numeric value of the concentration.
    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :return: Color corresponding to intensity of the concentration.
    """

    concentration_quantiles = get_percentile(concentration_type)
    first_quantile = concentration_quantiles[0]
    median = concentration_quantiles[1]
    third_quantile = concentration_quantiles[2]

    if np.isnan(concentration):  # Value not present
        return 'gray'
    elif math.isclose(concentration, 0):
        return '#ff99cc'    # Purple
    elif concentration <= first_quantile:
        return 'lightgreen'
    elif concentration <= median:
        return 'yellow'
    elif concentration <= third_quantile:
        return 'orange'
    # Bigger than third quantile will be red
    return 'red'

def populate_points(df, concentration_type):
    """ Fill a list with dictionary points.

    :param df: Chernobyl air concentration data frame
    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :return: List of dictionaries. Each dictionary contains data for a single point.
    """
    points = []
    for index in range(len(df)):
        single_point = fill_a_point(index, df, concentration_type)
        points.append(single_point)
    return points

def get_percentile(concentration_type):
    """ Get the first three quantiles for each concentration type.

    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :return: Array of quantile values corresponding to the concentration.
    """
    if concentration_type == 'i131':
        result = np.array([0.0034, 0.06, 1.14])
    elif concentration_type == 'cs134':
        result = np.array([0.0, 0.002035, 0.17])
    else:    # Expecting cs137
        result = np.array([0.0016, 0.02, 0.479325])

    return result

def populate_point_features(dict_points):
    """ Give plotting features to points. These points will be plotted on a folium map.

    :param dict_points: Dictionary of points containing Chernobyl air concentration data.
    :return: Dictionary of features to plot using geojson.
    """
    features = [
        {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': point['coordinates'],
            },
            'properties': {
                'time': point['time'],
                'icon': 'circle',
                'popup': point['popup'],
                'iconstyle': {
                    'fillColor': determine_color(point['concentration'], point['concentration_type']),
                    'fillOpacity': 0.6,
                    'stroke': 'false',
                    'radius': 5
                },
                'style': {'weight': 0}
            }
        } for point in dict_points
        ]
    return features
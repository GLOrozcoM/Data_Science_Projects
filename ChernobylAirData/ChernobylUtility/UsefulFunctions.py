import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from shapely.geometry import Point, Polygon
import geopandas as gpd

# TODO write test suites for each function

def read_world_map(shapefile):
    """ Take in a shape file, give the resulting geopandas data frame.

    shapefile -> geopandas.DataFrame
    """

    gdf_world = gpd.read_file(shapefile)[['ADMIN', 'ISO_A2', 'geometry']]
    gdf_world.columns = ['country', 'country_code', 'geometry']
    return gdf_world

# Test suite

def check_countries(dataFrameOne, dataFrameTwo):
    """ Check that both dataframes have the same countries based on country code. Smaller data frame should go first.

    prints a set difference if not the same
    Pandas dataframe1, Pandas dataframe2 -> boolean
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
    """ Erroneous country codes in original file. F -> France, UK -> GB.

    dataframe -> null
    """
    data_frame.loc[data_frame['country_code'] == 'F', 'country_code'] = 'FR'
    data_frame.loc[data_frame['country_code'] == 'UK', 'country_code'] = 'GB'
    return

    # Test suite


def set_norway_france(gdf_world):
    """ The world map provided by natural earth at a 50 mm resolution has erroneous entries for Norway and
    France. This function sets them straight.

    geo_data_frame -> Null
    """
    gdf_world.loc[gdf_world['country'] == 'France', 'country_code'] = 'FR'
    gdf_world.loc[gdf_world['country'] == 'Norway', 'country_code'] = 'NO'
    return

    # Test suite


def make_europe_map(df_chern, gdf_world):
    """ Cut the world map to only find countries affected by Chernobyl.

    data_frame (chernobyl data), geo_data_frame (world map) -> geo_pandas_data_frame containing european countries affected by Chernobyl
    """

    chern_countries = set(df_chern.country_code)

    europe_indices = []
    for index in range(len(gdf_world)):
        if gdf_world.country_code.values[index] in chern_countries:
            europe_indices.append(index)
    gdf_europe = gdf_world.iloc[europe_indices]

    gdf_europe = gdf_europe.append(gdf_world.loc[gdf_world.country == 'Ukraine'])

    gdf_europe.index = range(len(gdf_europe))

    return gdf_europe

# Test suite

def make_gdf_air(df_chern):
    """ Turn the chernobyl concentration data into a geo pandas data frame.

    data frame -> geopandas data_frame
    """

    gdf_air = gpd.GeoDataFrame(df_chern)
    crs = 'epsg:4326'
    geometry = [Point(xy) for xy in zip(df_chern["latitude"], df_chern["longitude"])]
    gdf_air = gpd.GeoDataFrame(df_chern, crs=crs, geometry=geometry)
    return gdf_air

# Test suite


def find_low_points(gdf):
    """ Takes a geo data frame and finds all points whose latitude coordinate is less than 22 degrees.

    geodataframe -> list of indices
    """
    index_counter = 0
    index_list = []
    for point in gdf.geometry.values:
        if point.xy[1][0] < 22:  # Access the latitude coordinate of a point
            index_list.append(index_counter)
        index_counter += 1
    return index_list

# Test suite

def modify_vaernes_valencia(gdf):
    """ Resets latitude and longitude coordinates for Vaernes and Valencia data. This includes a modification of the geometric column.

    gdf -> null
    """

    gdf.loc[gdf.city == 'VAERNES', 'latitude'] = 10.327740
    gdf.loc[gdf.city == 'VAERNES', 'longitude'] = 59.199860

    gdf.loc[gdf.city == 'VALENCIA', 'latitude'] = -0.376288
    gdf.loc[gdf.city == 'VALENCIA', 'longitude'] = 39.469906

    # Must modify since otherwise plotting would not work
    geometry = [Point(xy) for xy in zip(gdf["latitude"], gdf["longitude"])]
    gdf.geometry = geometry

    return gdf

# Test suite

def find_swapped_cz(gdf):
    """ This function finds the indices of entries in gdf that swapped their lats and longs for CZ.

    gdf -> [], indieces of swapped lats and longs
    """

    lat_range = [12, 19]
    long_range = [48.5, 51]

    gdf_cz = gdf.loc[gdf.country_code == 'CZ']
    out_indices = []
    for index in gdf_cz.index:
        current_lat = gdf.iloc[index].latitude
        current_long = gdf.iloc[index].longitude
        if current_lat < lat_range[0] or current_lat > lat_range[1]:
            if current_long < long_range[0] or current_long > long_range[1]:
                out_indices.append(index)
    return out_indices

# Test suite

def swap_lat_long(gdf, indices):
    """ Swap the latitude and longitudes of columns whose indices needed switching.

    gdf -> null
    """

    gdf['temp'] = gdf.loc[indices, 'latitude']
    gdf.loc[indices, 'latitude'] = gdf.loc[indices, 'longitude']
    gdf.loc[indices, 'longitude'] = gdf.loc[indices, 'temp']
    gdf.drop('temp', axis=1, inplace=True)

    geometry = [Point(xy) for xy in zip(gdf["latitude"], gdf["longitude"])]
    gdf.geometry = geometry

    return

# Test suite

def change_au(gdf_europe, gdf_world):
    """ Change data from Australia to be Austria's data from the world map.

    a gdf containing europe data -> null
    """

    au_index = gdf_europe.index[gdf_europe.country == 'Australia']
    au_index = au_index[0]  # Access the actual index value

    gdf_europe.at[au_index, 'country'] = 'Austria'
    gdf_europe.at[au_index, 'country_code'] = 'AT'
    gdf_europe.at[au_index, 'geometry'] = gdf_world.loc[gdf_world.country_code == 'AT', 'geometry'].values[0]

    return


def change_ir(gdf_europe, gdf_world):
    """ Change Iran's entry to be Ireland's in the european map.

    gdf -> null
    """
    ir_index = gdf_europe.index[gdf_europe.country_code == 'IR']
    ir_index = ir_index[0]

    gdf_europe = gdf_europe.drop(ir_index)
    gdf_europe = gdf_europe.append(gdf_world.loc[gdf_world.country == 'Ireland'])
    gdf_europe.index = range(len(gdf_europe))

    return gdf_europe

# Test suite

def change_slovakia(gdf_air, gdf_world):
    """ Take in the europe and world data sets. Change country code in gdf_air that should be in Slovakia.

    gdf_air, gdf_world -> gdf_air (modified)
    """
    slovakia_data = gdf_world.loc[gdf_world.country == 'Slovakia']

    for row_num in range(len(gdf_air)):
        point_interest = gdf_air.iloc[row_num].geometry
        if slovakia_data.geometry.contains(point_interest).values[0]:
            gdf_air.iat[row_num, 1] = 'SK'  # 1 to access country code
    return gdf_air

# Test suite

def add_slovakia(gdf_europe, gdf_world):
    """ Takes in the european map and adds in Slovakia from the world map.

    gdf_europe -> gdf_europe (modified)
    """

    slovakia_data = gdf_world.loc[gdf_world.country == 'Slovakia']
    gdf_europe = gdf_europe.append(slovakia_data)
    gdf_europe.index = range(len(gdf_europe))

    return gdf_europe

# TEst suite

def include_lux(gdf_europe, gdf_world):
    """ Take the european map data and add Luxembourg to it from the world data.

    gdf_europe, gdf_world -> gdf_europe (includes lux)
    """
    lux_data = gdf_world.loc[gdf_world.country_code == 'LU']
    gdf_europe = gdf_europe.append(lux_data)
    gdf_europe.index = range(len(gdf_europe))
    return gdf_europe

# Test suite

def fill_a_point(index, df):
    """ Fill up a point for the folium map using data for the ith index, and an air data frame.

    index, df -> Dictionary containing results.
    """
    coordinates = [df.loc[index, 'longitude'], df.loc[index, 'latitude']]

    begin_wrap = '<h1> '
    end_wrap = ' </h1>'
    city = df.loc[index, 'city']
    country_code = df.loc[index, 'country_code']
    popup = begin_wrap + city + ', ' + country_code + end_wrap

    time = df.loc[index, 'Date']

    result = {'coordinates': coordinates, 'popup': popup, 'time': time}
    return result


def populate_points(df):
    """ Fill up the points list with points following the fill a point format.

    df -> list of points
    """
    points = []
    for index in range(len(df)):
        single_point = fill_a_point(index, df)
        points.append(single_point)
    return points

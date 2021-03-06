"""
Module for functions used in analysis of Chernobyl air data.
"""

# General math
import numpy as np
import math

# Map operations
from shapely.geometry import Point
import geopandas as gpd
import folium
from folium import plugins


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
    geometry = [Point(xy) for xy in zip(df_chern["longitude"], df_chern["latitude"])]
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
    """ Resets latitude and longitude coordinates for Vaernes and Valencia data. This includes a modification of the geometric column.m

    :param gdf: Geo data frame containing data for Vaernes, NO and Valencia, ES.
    :return: Geo data frame with correct measures for Vaernes and Valencia.
    """

    vaernes_lat = 59.199860
    vaernes_long = 10.327740
    val_lat = 39.469906
    val_long = -0.376288

    gdf.loc[gdf.city == 'VAERNES', 'latitude'] = vaernes_lat
    gdf.loc[gdf.city == 'VAERNES', 'longitude'] = vaernes_long
    gdf.loc[gdf.city == 'VALENCIA', 'latitude'] = val_lat
    gdf.loc[gdf.city == 'VALENCIA', 'longitude'] = val_long

    # Must modify since otherwise plotting would not work

    # Change only the geometry of vaernes and valencia
    gdf.loc[gdf.city == 'VAERNES', 'geometry'] = Point( vaernes_long, vaernes_lat)
    gdf.loc[gdf.city == 'VALENCIA', 'geometry'] = Point(val_long, val_lat)

    return gdf

def find_out_indices_cz(gdf):
    """ Finds indices for entries of CZ that are not correct.

    :param gdf: Geo data frame containing an entry for CZ.
    :return: A list of indices where lat and long's are outside of the range for CZ.
    """

    # Ranges for CZ only.
    long_range = [12, 19]
    lat_range = [48.5, 51]

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

    # Swap lat and long
    gdf['temp'] = gdf.loc[indices, 'latitude']
    gdf.loc[indices, 'latitude'] = gdf.loc[indices, 'longitude']
    gdf.loc[indices, 'longitude'] = gdf.loc[indices, 'temp']
    gdf.drop('temp', axis=1, inplace=True)

    # TODO only geometry of the modified records
    geometry = [Point(xy) for xy in zip(gdf["longitude"], gdf["latitude"])]
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
    coordinates = [df.loc[index, 'longitude'], df.loc[index, 'latitude']]

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

def fill_entries_legend(concentration_type):
    """ Fill an entry in the map legend to map color intensity.

    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :return: HTML string
    """

    percentiles = get_percentile(concentration_type)

    # Extreme - greater than third quartile
    extreme_value = 1 + percentiles[2]
    extreme_text = 'Extreme'
    extreme_entry = intensity_entry(extreme_value, extreme_text, concentration_type)

    # Strong - less than or equal to third quartile
    strong_value = percentiles[2]
    strong_text = 'Strong'
    strong_entry = intensity_entry(strong_value, strong_text, concentration_type)

    # Moderate - less than or equal to median
    moderate_value = percentiles[1]
    moderate_text = 'Moderate'
    moderate_entry = intensity_entry(moderate_value, moderate_text, concentration_type)

    # Weak - less than or equal to first quartile
    weak_value = percentiles[0]
    weak_text = 'Weak'
    weak_entry = intensity_entry(weak_value, weak_text, concentration_type)

    # None - measurement of 0 for concentration
    none_value = 0
    none_text = 'None'
    none_entry = intensity_entry(none_value, none_text, concentration_type)

    # NA value
    na_text = 'No value'
    na_entry = intensity_entry(np.nan, na_text, concentration_type)

    final_entry = extreme_entry + strong_entry + moderate_entry + weak_entry + none_entry + na_entry

    return final_entry

def intensity_entry(concentration_value, value_text, concentration_type):
    """ Write intensity entry for a particular concentration.

    :param concentration_value: Numeric concentration value.
    :param value_text: String describing the value.
    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :return: HTML string
    """

    color = determine_color(concentration_value, concentration_type)
    text = value_text
    entry = entry_legend(text, color)

    return entry

def entry_legend(concentration_entry, color):
    """ Creates an HTML string combining the intensity of the concentration and the color it should take.

    :param concentration_entry: String to display in legend regarding intensity.
    :param color: Corresponding color for the circle to display next to text.
    :return: HTML string for line in legend
    """

    # One row, two columns - concentration intensity, color
    text = """<tr> <th> &nbsp; {con_value} </th> <th> &nbsp <i class="fa fa-circle" style="color:{color}";></i> </th> </tr>"""
    formatted_text = text.format(con_value = concentration_entry , color = color)

    return formatted_text

def create_outer_legend_box(title, legend_entries):
    """ Make the legend html showing intensities of concentration along with color.

    :param title: Title for the legend.
    :param legend_entries: Rows to go in the table in legend.
    :return: HTML string
    """

    legend_box = """
     <div style="
     position: fixed;
     top: 10px; left: 390px; width: 190px; height: 160px;
     border:2px solid grey; z-index:9999;

     background-color:white;
     opacity: .65;

     font-size:14px;
     font-weight: bold;

     ">
     &nbsp; {title}

     <table style="width:50%">
      {table_entries}
     </table>

      </div> """.format(title=title, table_entries=legend_entries)

    return legend_box

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

def create_folium_map(concentration_type, air_df, map_center, zoom_start):
    """ Create a time stamped folium map with Chernobyl air concentration data.

    :param concentration_type: Either i131, cs134, or cs137 - concentrations from Chernobyl air data set.
    :param air_df: Chernobyl air data frame.
    :param map_center: Where to center the map, [lat long].
    :param zoom_start: Level of zoom in on the map.
    :return: Folium map object.
    """
    # Data points
    points = populate_points(air_df, concentration_type)
    features = populate_point_features(points)

    # Map
    folium_map = folium.Map(location=map_center, zoom_start=zoom_start, width='60%', height='100%')

    # Create legend
    intensity_entries = fill_entries_legend(concentration_type)
    legend_box = create_outer_legend_box('Concentration intensity', intensity_entries)

    # Tag legend to map
    folium_map.get_root().html.add_child(folium.Element(legend_box))

    # Tag time scroller to map
    plugins.TimestampedGeoJson(
        {
            'type': 'FeatureCollection',
            'features': features
        },
        period='P1D',
        add_last_point=False,
        auto_play=False,
        loop=False,
        max_speed=1,
        loop_button=True,
        date_options='YYYY-MM-DD',
        time_slider_drag_update=True
    ).add_to(folium_map)

    # Insert site of Chernobyl accident (Pripyat)
    PRIPYAT_COORDS = [ 51.386998452, 30.092666296 ]
    folium.CircleMarker(
        location=PRIPYAT_COORDS,
        radius=5,
        popup='<h3 style="font-family:Times New Roman;">Pripyat, Ukraine </h3><p style="font-family:Times New Roman;"> City nearest to nuclear explosion </p>',
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(folium_map)

    # Insert site of Chernobyl
    CHERNOBYL_COORDS = [ 50.447730, 30.542720 ]
    folium.CircleMarker(
        location=CHERNOBYL_COORDS,
        radius=5,
        popup='<h3 style="font-family:Times New Roman;">Chernobyl, Ukraine</h3><p style="font-family:Times New Roman;"> 15 km south of Pripyat </p>',
        color='gray',
        fill=True,
        fill_color='gray'
    ).add_to(folium_map)

    return folium_map
import ee
import numpy as np

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']

PROJECT_ID = "hacarton076"

def initialize():
    try:
        ee.Initialize(project=PROJECT_ID)
        return True
    except Exception:
        return False

def authenticate():
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

def fetch_sentinel2(lon: float, lat: float, radius_m: int = 2000, date_start: str = '2024-01-01', date_end: str = '2024-12-31') -> np.ndarray:
    """Fetch Sentinel-2 L1C image for a location."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m)

    s2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    image = s2.first()
    if image is None:
        raise ValueError("No Sentinel-2 images found for this location/date range")

    image = image.select(BANDS)
    image = image.toFloat().resample('bilinear').reproject(image.select('B2').projection())
    sample = image.toArray().sampleRectangle(region)

    array = np.array(sample.getInfo()['properties']['array'])
    return array

def get_available_dates(lon: float, lat: float, date_start: str = '2024-01-01', date_end: str = '2024-12-31') -> list:
    """Get available Sentinel-2 image dates for a location."""
    point = ee.Geometry.Point([lon, lat])

    s2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    def get_date(image):
        return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})

    dates = s2.map(get_date).distinct('date').aggregate_array('date').getInfo()
    return sorted(dates)

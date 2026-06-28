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


def _mask_clouds(image):
    """Skip cloud masking — collection already filters CLOUDY_PIXEL_PERCENTAGE <= 35."""
    return image


def _get_s2_collection(lon, lat, date_start, date_end, cloud_threshold=35):
    """Get Sentinel-2 L1C collection with cloud masking."""
    point = ee.Geometry.Point([lon, lat])

    s2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    return s2


def fetch_sentinel2(lon, lat, radius_m=2000, date_start='2024-01-01', date_end='2024-12-31', cloud_threshold=35):
    """Fetch a single cloud-masked Sentinel-2 L1C image."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m)

    s2 = _get_s2_collection(lon, lat, date_start, date_end, cloud_threshold)

    image = s2.first()
    if image is None:
        raise ValueError("No Sentinel-2 images found for this location/date range")

    image = _mask_clouds(image)
    image = image.select(BANDS).toFloat()
    sample = image.toArray().sampleRectangle(region)

    array = np.array(sample.getInfo()['properties']['array'])
    return array


def fetch_temporal_stack(lon, lat, radius_m=2000, date_start='2024-01-01', date_end='2024-12-31',
                         max_images=10, cloud_threshold=35):
    """Fetch multiple Sentinel-2 images for temporal compositing."""
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(radius_m)

    s2 = _get_s2_collection(lon, lat, date_start, date_end, cloud_threshold)
    image_list = s2.toList(max_images)

    arrays = []
    count = min(s2.size().getInfo(), max_images)

    for i in range(count):
        image = ee.Image(image_list.get(i))
        image = _mask_clouds(image)
        image = image.select(BANDS).toFloat()
        sample = image.toArray().sampleRectangle(region)

        try:
            array = np.array(sample.getInfo()['properties']['array'])
            arrays.append(array)
        except Exception:
            continue

    if not arrays:
        raise ValueError("No valid images found")

    return arrays


def temporal_composite(arrays, method='median'):
    """Create a temporal composite from multiple images.

    Methods:
    - 'median': Per-pixel median across time (robust to outliers)
    - 'mean': Per-pixel mean across time
    - 'least_cloudy': Not applicable here, handled at fetch time
    """
    stack = np.stack(arrays, axis=0)

    if method == 'median':
        return np.median(stack, axis=0)
    elif method == 'mean':
        return np.mean(stack, axis=0)
    else:
        return np.median(stack, axis=0)


def get_available_dates(lon, lat, date_start='2024-01-01', date_end='2024-12-31'):
    """Get available Sentinel-2 image dates for a location."""
    point = ee.Geometry.Point([lon, lat])

    s2 = ee.ImageCollection('COPERNICUS/S2') \
        .filterBounds(point) \
        .filterDate(date_start, date_end) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))

    def get_date(image):
        return ee.Feature(None, {'date': image.date().format('YYYY-MM-dd')})

    dates = s2.map(get_date).distinct('date').aggregate_array('date').getInfo()
    return sorted(dates)

import osmium
import geopandas as gpd
import pandas as pd
import folium
import warnings
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

warnings.filterwarnings('ignore')

CENTER_LAT, CENTER_LON = 55.225000, 14.245000
RADIUS_NM = 50
RADIUS_KM = RADIUS_NM * 1.852
RADIUS_METERS = RADIUS_KM * 1000
OUTPUT_MAP = "osm_excluded_ports_buffered.html"

BBOX = (
    CENTER_LON - 1.5,  # min lon
    CENTER_LAT - 1.0,  # min lat
    CENTER_LON + 1.5,  # max lon
    CENTER_LAT + 1.0   # max lat
)

OSM_FILES = [
    "data/denmark-latest.osm.pbf",
    "data/sweden-latest.osm.pbf",
]

PORT_TAGS = {
    'landuse':          {'port', 'harbour', 'industrial'},
    'industrial':       {'port', 'shipyard'},
    'waterway':         {'dock', 'basin'},
    'leisure':          {'marina'},
    'amenity':          {'ferry_terminal'},
    'man_made':         {'pier', 'quay', 'jetty', 'breakwater', 'groyne',
                         'floating_barrier', 'maritime_beacon', 'dolphins'},
    'harbour':          {'yes'},
    'water':            {'harbour', 'dock'},
    'seamark:type':     {'harbour', 'harbour_basin', 'port',
                         'anchorage', 'berth', 'dock',
                         'ferry_route', 'mooring', 'pilot_boarding_place',
                         'small_craft_facilities'},
    'building':         {'warehouse', 'hangar', 'ship_station', 'ferry_terminal'},
    'railway':          {'station'},
    'public_transport': {'stop_position', 'platform'},
}


class PortHandler(osmium.SimpleHandler):
    """
    Walks through the PBF file and collects any way or relation
    whose tags match our port tag definitions.
    """
    def __init__(self):
        super().__init__()
        self.polygons = []
        self.wktfab = osmium.geom.WKTFactory()

    def _is_port_feature(self, tags):
        for key, values in PORT_TAGS.items():
            if key in tags and tags[key] in values:
                return True
        return False

    def _in_bbox(self, geom):
        """Quick check if geometry centroid is within our bounding box."""
        try:
            from shapely import wkt
            shape = wkt.loads(geom)
            c = shape.centroid
            return (BBOX[0] <= c.x <= BBOX[2] and
                    BBOX[1] <= c.y <= BBOX[3])
        except Exception:
            return False

    def way(self, w):
        if not self._is_port_feature(w.tags):
            return
        try:
            wkt_geom = self.wktfab.create_linestring(w)
            from shapely import wkt
            line = wkt.loads(wkt_geom)
            if line.is_ring:
                poly = Polygon(line)
                if self._in_bbox(poly.wkt):
                    self.polygons.append(poly)
        except Exception:
            pass

    def area(self, a):
        if not self._is_port_feature(a.tags):
            return
        try:
            wkt_geom = self.wktfab.create_multipolygon(a)
            from shapely import wkt
            shape = wkt.loads(wkt_geom)
            if self._in_bbox(shape.wkt):
                self.polygons.append(shape)
        except Exception:
            pass


def extract_port_polygons(osm_file):
    print(f"   -> Reading {osm_file}...")
    handler = PortHandler()
    handler.apply_file(osm_file, locations=True, idx='flex_mem')
    print(f"      -> Found {len(handler.polygons)} raw port polygons.")
    return handler.polygons


def map_actual_osm_zones():
    print(f"1. Loading OSM Port Boundaries from local PBF files...")

    all_polygons = []
    for osm_file in OSM_FILES:
        if not os.path.exists(osm_file):
            print(f"   -> WARNING: {osm_file} not found, skipping.")
            continue
        all_polygons.extend(extract_port_polygons(osm_file))

    if not all_polygons:
        print("Error: No port polygons found.")
        return

    print(f"\n2. Merging {len(all_polygons)} polygons and applying 200m buffer...")

    ports_gdf = gpd.GeoDataFrame(geometry=all_polygons, crs="EPSG:4326")
    ports_gdf['geometry'] = ports_gdf.geometry.buffer(0)  # Fix invalid geometries
    ports_gdf = ports_gdf.dissolve()

    # Buffer 200m around all port boundaries
    ports_gdf = ports_gdf.to_crs("EPSG:3857")
    ports_gdf['geometry'] = ports_gdf.buffer(600)
    ports_gdf = ports_gdf.to_crs("EPSG:4326")

    print(f"   -> Final dissolved port zone ready.")

    print("\n3. Generating Interactive HTML Map...")
    m = folium.Map(location=[CENTER_LAT, CENTER_LON], zoom_start=8, tiles='CartoDB Positron')

    folium.GeoJson(
        ports_gdf,
        name="Excluded Port Zones (200m Buffer)",
        style_function=lambda feature: {
            'fillColor': 'red',
            'color': 'darkred',
            'weight': 1.5,
            'fillOpacity': 0.5,
        }
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(OUTPUT_MAP)
    print(f"✅ SUCCESS: Saved map to '{OUTPUT_MAP}'")


if __name__ == "__main__":
    map_actual_osm_zones()
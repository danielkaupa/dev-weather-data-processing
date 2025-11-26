import geopandas as gpd
import matplotlib.pyplot as plt

# Path to your extracted GeoJSON
INDIA_GEOJSON_PATH = "geoBoundariesCGAZ_ADM0/india_boundary.geojson"

def main():
    # Load geojson
    gdf = gpd.read_file(INDIA_GEOJSON_PATH)

    print("CRS:", gdf.crs)
    print("Number of features:", len(gdf))
    print("Geometry type(s):", gdf.geom_type.unique())

    # Basic plot
    fig, ax = plt.subplots(figsize=(8, 10))
    gdf.plot(ax=ax, edgecolor="black", facecolor="lightblue")

    ax.set_title("Extracted India Boundary (Quick Verification)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.show()


if __name__ == "__main__":
    main()

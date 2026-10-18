from matplotlib import pyplot as plt
import pandas as pd

def plot_files(csv_files):
    plt.figure()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[df['value'] != 0]
        smoothed = df['count'].rolling(window=100, center=True, min_periods=1).mean()
        plt.plot(df['value'], smoothed, label=csv_file.split('_')[0])
    plt.title('Smoothed Elevation Count Distribution')
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.xlim(0, 7000)
    plt.legend()
    #plt.show()
    plt.savefig('smoothed_elevation_count_distribution.pdf')
    plt.close()

def plot_probability_density(csv_files):
    plt.figure()
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df[df['value'] != 0]
        smoothed = df['count'].rolling(window=100, center=True, min_periods=1).mean()
        plt.plot(df['value'], smoothed / smoothed.sum(), label=csv_file.split('_')[0])
    plt.title('Probability Density Function')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.xlim(0, 7000)
    plt.legend()
    #plt.show()
    plt.savefig('probability_density_function.pdf')
    plt.close()

plot_files(["africa_raster_table.csv", "antarctica_raster_table.csv", "asia_raster_table.csv", "australia_raster_table.csv", "europe_raster_table.csv", "north_america_raster_table.csv", "south_america_raster_table.csv", "oceania_raster_table.csv"])
plot_probability_density(["africa_raster_table.csv", "antarctica_raster_table.csv", "asia_raster_table.csv", "australia_raster_table.csv", "europe_raster_table.csv", "north_america_raster_table.csv", "south_america_raster_table.csv", "oceania_raster_table.csv"])
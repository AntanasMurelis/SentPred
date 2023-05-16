import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import umap_ as umap
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import imageio
import os
from scipy.ndimage import gaussian_filter1d
from datetime import datetime

def visualize_embeddings(embeddings, sentiment_labels, layer = None, method='pca', random_state=42, sample_size=None, ax = None):
    if sample_size is not None and sample_size < len(embeddings):
        sentiment_labels = np.array(sentiment_labels)
        unique_labels = np.unique(sentiment_labels)
        sample_size_per_label = sample_size // len(unique_labels)
        idx = []

        for label in unique_labels:
            label_idx = np.where(sentiment_labels == label)[0]
            label_sample_idx = np.random.choice(label_idx, sample_size_per_label, replace=False)
            idx.extend(label_sample_idx)

        np.random.shuffle(idx)
        embeddings = embeddings[idx]
        sentiment_labels = sentiment_labels[idx]

    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=random_state, early_exaggeration=100, perplexity = 30, init = "random")
    elif method.lower() == 'umap':
        reducer = umap.UMAP(random_state=random_state)
    else:
        raise ValueError("Invalid method. Choose from 'pca', 'tsne', or 'umap'")

    reduced_embeddings = reducer.fit_transform(embeddings)

    scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], s=2, c=sentiment_labels, cmap='viridis', alpha=0.8)
    ax.set_title(f"{method.upper()} Visualization of Tweet Embeddings (Layer unfrozen {layer})")

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    return scatter

def make_US_map_gif(df, shapefile_path):

    # Create the directory if it doesn't exist
    if not os.path.exists('maps'):
        os.makedirs('maps')

    # Order by date
    df = df.sort_values(by='Timestamp')

    # Aggregate by state and date to calculate daily mean sentiment for each state
    daily_sentiment = df.groupby([df['Timestamp'].dt.date, 'state'])['Sentiment_Class'].mean().reset_index()

    # Convert the dates to datetime format
    daily_sentiment['Timestamp'] = pd.to_datetime(daily_sentiment['Timestamp'])

    # Create a grid of all dates and all states
    all_dates = pd.date_range(start=daily_sentiment['Timestamp'].min(), end=daily_sentiment['Timestamp'].max())
    all_states = daily_sentiment['state'].unique()
    grid = pd.MultiIndex.from_product([all_dates, all_states], names=['Timestamp', 'state']).to_frame(index=False)

    # Merge the original data with the grid to fill in missing data
    daily_sentiment = pd.merge(grid, daily_sentiment, on=['Timestamp', 'state'], how='left')

    # Impute NaN values with 0
    daily_sentiment = daily_sentiment.fillna(0)

    # Convert the datetime values to Unix timestamp
    daily_sentiment['Timestamp'] = daily_sentiment['Timestamp'].astype(int) / 10**9

    # Define your Gaussian kernel
    sigma = 5  # you may need to adjust this value depending on your needs

    # Apply a Gaussian kernel by smoothing the 'Average_Sentiment' values
    daily_sentiment['Sentiment_Class'] = gaussian_filter1d(daily_sentiment['Sentiment_Class'], sigma)

    # Load a GeoDataFrame with the geometry of each state
    us_states = gpd.read_file(shapefile_path)

    # Filter out non-continental US states and territories
    non_continental = ['Hawaii', 'Alaska', 'Puerto Rico', 'Guam', 'American Samoa', 'U.S. Virgin Islands', 'Northern Mariana Islands']
    us_states = us_states[~us_states['NAME'].isin(non_continental)]

    # Normalize the sentiment scores to a range that fits your colormap
    vmin, vmax = -1, 1
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Prepare a list to store the filepaths for the images
    filepaths = []

    for date, sentiment in daily_sentiment.groupby('Timestamp'):
        # Left merge the sentiment scores with the GeoDataFrame, keeping all states even if they don't have sentiment data
        sentiment_map = us_states.merge(sentiment, how='left', left_on='NAME', right_on='state')

        # Create the map
        fig, ax = plt.subplots(figsize=(15, 10))
        fig.tight_layout()
        ax.set_axis_off()

        # Create a map and a colorbar
        sentiment_plot = sentiment_map.plot(column='Sentiment_Class', cmap='RdBu', linewidth=0.1, ax=ax, edgecolor='k', legend=False, norm=norm)
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu'), ax=ax, orientation='vertical', fraction=0.0325, pad=0.1)
        # Set the map limits to be consistent across all maps
        ax.set_xlim(-130, -60)
        ax.set_ylim(24, 50)

        # Convert the Unix timestamp back to a datetime object
        date_datetime = datetime.fromtimestamp(date)

        plt.title(f'Sentiment Map {date_datetime.strftime("%Y-%m-%d")}')
        
        # Save the figure
        filepath = f'maps/map_{date_datetime.strftime("%Y-%m-%d")}.png'
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

        # Append the filepath to the list
        filepaths.append(filepath)

    # Use imageio to compile the images into a gif
    images = [imageio.imread(filepath) for filepath in filepaths]
    imageio.mimsave('sentiment_map.gif', images, duration=0.2)



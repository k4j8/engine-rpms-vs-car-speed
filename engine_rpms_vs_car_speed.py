#!/bin/python3

"""Plots engine RPMs against car speed and performs clustering to determine
gear number
"""

# pylint: disable=invalid-name

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# Define source data (required)
SOURCES = ['rpms_chart_from_obd.csv']

# Define column names from data (required)
SPEED = 'Speed (OBD)(mph)'
ENGINE = 'Engine RPM(rpm)'

# Define parameters for removing outliers (can leave at defaults below)
GEARS = 5
MIN_SPEED = 5
MIN_ENGINE = 900
MAX_ERROR = 1
PERCTILE_ERROR_REMOVAL = 2


def mc(x, y):
    """Calculate ordinary least squares parameters
    Code was taken from:
    https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    """

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]

    return m, c


def ols(x, y):
    """Calculate OLS line"""

    m, c = mc(x, y)

    return m * x + c


def remove_outliers(df, direction, percentile_target, error):
    """Remove outliers"""

    percentile_value = np.percentile(df['distance from best-fit'],
                                     percentile_target)
    rows_before = len(df)
    if direction == 'below':
        if percentile_value <= error:
            df.drop(df[df['distance from best-fit'] < percentile_value].index,
                    inplace=True)
    elif direction == 'above':
        if percentile_value >= error:
            df.drop(df[df['distance from best-fit'] > percentile_value].index,
                    inplace=True)

    rows_removed = len(df) != rows_before

    return df, rows_removed


def update_figure_layout(fig, legend):
    """Updates figure and axis titles"""

    fig.update_layout(title_text='Engine RPMs vs. Car Speed',
                      showlegend=legend)
    fig.update_xaxes(title='Car Speed (mph)', range=[0, 100])
    fig.update_yaxes(title='Engine (RPMs)', range=[0, 6000])
    fig.show()

    return fig


def main():
    """Import data, perform clustering, then plot result"""

    # Import data
    df_list = [pd.read_csv(source, usecols=[SPEED, ENGINE])
               for source in SOURCES]
    df = pd.concat(df_list)

    # Calculate ratio of RPMs to mph
    df['ratio'] = df[ENGINE] / df[SPEED]

    # Drop rows where ratio is infinity
    df.replace(to_replace=np.inf, value=np.nan, inplace=True)
    df.dropna(subset='ratio', inplace=True)

    # Create folder for images
    if not os.path.exists('images'):
        os.mkdir('images')

    # Graph raw data
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
                             x=df[SPEED],
                             y=df[ENGINE],
                             mode='markers',
                            ))
    update_figure_layout(fig1, False)
    fig1.write_image(os.path.join('images',
        'Engine RPMs vs. Car Speed without clustering.png'))

    # Drop rows where speed is less than MIN_SPEED
    df.drop(df[df[SPEED] < MIN_SPEED].index, inplace=True)

    # Drop rows where engine is less than MIN_ENGINE
    df.drop(df[df[ENGINE] < MIN_ENGINE].index, inplace=True)

    # K-Means
    data_1d = df['ratio'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=GEARS).fit(data_1d)
    df['label'] = kmeans.labels_

    # Graph with clustering and removing outliers
    centers = kmeans.cluster_centers_[:, 0]  # centers of each cluster
    clusters = np.flipud(np.argsort(centers))  # rank each center from highest down

    fig2 = go.Figure()
    for i, cluster in enumerate(clusters):
        gear_number = i + 1
        df_cluster = df[df['label'] == cluster].copy()  # get df for this cluster only

        speed = df_cluster[SPEED]
        engine = df_cluster[ENGINE]
        removing_low = True
        removing_high = True
        while removing_low or removing_high:
            # Remove outliers
            m, c = mc(speed, engine)
            df_cluster['distance from best-fit'] = (abs(m * speed - engine + c)
                                                    / (m**2 + 1)**0.5)

            df_cluster, removing_low = remove_outliers(df_cluster,
                                                       'below',
                                                       PERCTILE_ERROR_REMOVAL,
                                                       -1 * MAX_ERROR)
            df_cluster, removing_high = remove_outliers(df_cluster,
                                                        'above',
                                                        100 - PERCTILE_ERROR_REMOVAL,
                                                        MAX_ERROR)
            speed = df_cluster[SPEED]
            engine = df_cluster[ENGINE]

        # Add cluster
        fig2.add_trace(go.Scatter(
                                  x=speed,
                                  y=engine,
                                  mode='markers',
                                  name=f'Gear {gear_number}',
                                 ))

        # Add best-fit line for cluster
        fig2.add_trace(go.Scatter(
                                  x=speed,
                                  y=ols(speed, engine),
                                  mode='lines',
                                  marker_color='black',
                                  name=f'Best-Fit for Gear {gear_number}: {m:.0f}*x{c:+.0f}',
                                 ))

    fig2 = update_figure_layout(fig2, True)
    fig2.write_image(os.path.join('images',
        'Engine RPMs vs. Car Speed with clustering.png'))


if __name__ == '__main__':
    main()

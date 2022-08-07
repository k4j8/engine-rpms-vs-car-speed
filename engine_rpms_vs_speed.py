#!/bin/python3

"""Plots engine RPMs against car speed and performs clustering to deterimen
gear number
"""

# pylint: disable=invalid-name

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from sklearn.cluster import KMeans

# Define source data
SOURCE = 'rpms_chart_from_obd.xlsx'

# Define column names from data
SPEED = 'Speed (OBD)(mph)'
ENGINE = ' Engine RPM(rpm)'

# Define parameters for removing outliers
MAX_ERROR = 140
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


def main():
    """Import data, perform clustering, then plot result"""

    # Import data
    df = pd.read_excel(SOURCE, usecols=[SPEED, ENGINE])

    # Calculate ratio of RPMs to mph
    df['ratio'] = df[ENGINE] / df[SPEED]

    # Drop rows where ratio is infinity
    df.replace(to_replace=np.inf, value=np.nan, inplace=True)
    df.dropna(subset='ratio', inplace=True)

    # Drop rows where speed is less than 5
    df.drop(df[df[SPEED] < 5].index, inplace=True)

    # K-Means
    data_1d = df['ratio'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=5).fit(data_1d)
    df['label'] = kmeans.labels_

    # Graph
    centers = kmeans.cluster_centers_[:, 0]  # centers of each cluster
    clusters = np.flipud(np.argsort(centers))  # rank each center from highest down

    fig = go.Figure()
    for i, cluster in enumerate(clusters):
        gear_number = i + 1
        df_for_label = df[df['label'] == cluster].copy()  # get df for this cluster only

        speed = df_for_label[SPEED]
        engine = df_for_label[ENGINE]
        removing_low = True
        removing_high = True
        while removing_low or removing_high:
            m, c = mc(speed, engine)
            df_for_label['distance from best-fit'] = engine - m * speed + c  # TODO use Euclidean distance

            df_for_label, removing_low = remove_outliers(df_for_label,
                                                         'below',
                                                         PERCTILE_ERROR_REMOVAL,
                                                         -1 * MAX_ERROR)
            df_for_label, removing_high = remove_outliers(df_for_label,
                                                          'above',
                                                          100 - PERCTILE_ERROR_REMOVAL,
                                                          MAX_ERROR)

            speed = df_for_label[SPEED]
            engine = df_for_label[ENGINE]

        # Add cluster
        fig.add_trace(go.Scatter(
                                 x=speed,
                                 y=engine,
                                 mode='markers',
                                 name=f'Gear {gear_number}',
                                ))

        # Add best-fit line for cluster
        fig.add_trace(go.Scatter(
                                 x=df_for_label[SPEED],
                                 y=ols(speed, engine),
                                 mode='lines',
                                 marker_color='black',
                                 name=f'Best-Fit for Gear {gear_number}',
                                ))

    # Update titles
    fig.update_layout(title_text='RPMs vs mph', showlegend=True)
    fig.update_xaxes(title='Speed (mph)')
    fig.update_yaxes(title='Engine (RPMs)')
    fig.show()


if __name__ == '__main__':
    main()

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gating_choices_plot(choices, gate_names, title, xlabel):
    plt.hist(tf.where(choices).numpy()[:, 1], bins=100)
    plt.xticks(np.arange(len(gate_names)), labels=gate_names, rotation=90)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Count')
    plt.show()

def routing_plot(choices, route_names, opt=''):
    gating_choices_plot(
        choices=choices,
        gate_names=route_names,
        title='Which routes were taken by data? ' + opt,
        xlabel='Route'
    )

def routing_summary(choices, route_names):
    n = choices.shape[0]
    summed = tf.reduce_sum(choices, axis=0).numpy()
    df = pd.DataFrame(columns=route_names)
    df = df.T
    df['count'] = summed
    df['percentage'] = summed / n
    return df

def plot_n_routes(n_routes, avg, std):
    n, _, _ = plt.hist(n_routes, bins=100)
    ycoord = np.max(n) + 100
    plt.title('How many routes are taken?')
    plt.xlabel('Number of sub-networks used')
    plt.xticks(np.arange(np.max(n_routes)+1))
    plt.ylabel('Count')
    plt.vlines(avg, 0, ycoord, colors='red', linestyles='dashed', label=f'Average (= {avg:.2f})')
    plt.text(avg-0.4, ycoord+10, f'{avg:.2f}', color='red')
    eb = plt.errorbar(x=avg, y=(ycoord/2), xerr=std, color='green', label=f'Standard deviation (= {std:.2f})', capsize=4)
    eb[-1][0].set_linestyle('--')
    plt.legend()
    plt.show()

def route_weights(r_weights, feature_types):
    df = pd.DataFrame(columns=feature_types)
    df = df.T
    df['mean'] = np.mean(r_weights, axis=0)
    df['std'] = np.std(r_weights, axis=0)
    return df

def plot_route_weights(df):
    plt.bar(x=np.arange(len(df.index)), height=df['mean'], yerr=df['std'])
    plt.xticks(np.arange(len(df.index)), labels=df.index, rotation=90)
    plt.title('Route weights')
    plt.ylabel('Average + standard deviation')
    plt.xlabel('Route')
    plt.show()

def n_routes_df(n_routes):
    df = pd.DataFrame(columns=np.arange(np.max(n_routes)+1))
    df = df.T
    df['count'] = [np.count_nonzero(n_routes == i) for i in df.index]
    df['%'] = df['count'] / n_routes.shape[0]
    return df

def fgating_hist(masks, title_suffix=''):
    n_ftrs = tf.reduce_sum(masks, axis=-1)
    avg = np.mean(n_ftrs)
    std = np.std(n_ftrs)
    
    n, _, _ = plt.hist(n_ftrs, bins=100)
    ymax = np.max(n)
    plt.title('Number of activated features ' + title_suffix)
    plt.xlabel('Number of activated features')
    plt.ylabel('Count')
    plt.axvline(avg, color='red', linestyle='dashed', label='Average')
    plt.text(avg*1.005, plt.ylim()[1]*0.9, 'Average: {:.2f}'.format(avg), color='red')
    eb = plt.errorbar(x=avg, y=(ymax/2), xerr=std, color='green', label=f'Standard deviation (= {std:.2f})', capsize=4)
    eb[-1][0].set_linestyle('--')
    plt.legend()
    plt.show()

def fgating_topftrs(masks, feature_names):
    # Reduce over feature
    f_activated = tf.reduce_sum(masks, axis=0)
    # Which features are activated ?
    f_activations = [(feature_names[i], f_activated[i].numpy()) for i in tf.where(f_activated)[:, 0]]
    top_features = sorted(f_activations, key = lambda x : x[1], reverse=True)
    # top_features = [x for x, _ in top_features]
    return top_features

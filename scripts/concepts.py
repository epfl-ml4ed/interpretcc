# Code adapted and taken from https://github.com/epfl-ml4ed/ripple/blob/main/scripts/tcav_2.ipynb
# Acknowledgements to Mohamad Asadi

import ast
import numpy as np
import pandas as pd

def give_top_bot_percent(input_series, n):
    series = input_series[input_series != 0]
    return series.iloc[np.argsort(series).iloc[-int(len(series)*n):]], series.iloc[np.argsort(series).iloc[:int(len(series)*n)]]

def effort(features_df, n):
    '''Effort: Higher/Lower intensity'''
    # Top: Intersection
    # Bot: Union
    effort_df = features_df[['user_id', 'time_sessions sum', 'total_clicks_video']]
    
    top_time_sessions_sum, bot_time_sessions_sum = give_top_bot_percent(effort_df['time_sessions sum'].apply(sum), n)
    top_total_video_clicks, bot_total_video_clicks = give_top_bot_percent(effort_df['total_clicks_video'].apply(sum), n)
    
    top_intersection = np.array(list(set(top_time_sessions_sum.index).intersection(top_total_video_clicks.index)))
    bot_intersection = np.array(list(set(bot_time_sessions_sum.index).union(bot_total_video_clicks.index)))
    
    lim = np.min([len(top_intersection), len(bot_intersection)])
    
    top_intersection = top_intersection[np.random.choice(len(top_intersection), size=(lim,), replace=False)] if len(top_intersection) > lim else top_intersection
    bot_intersection = bot_intersection[np.random.choice(len(bot_intersection), size=(lim,), replace=False)] if len(bot_intersection) > lim else bot_intersection
    
    return top_intersection, bot_intersection

def consistency(features_df, n, add_mean_session = True):
    '''Consistency: Uniform / First half / Second half'''
    add_mean_session = True
    consistency_df = features_df[['relative_time_online', 'relative_video_clicks', 'time_sessions mean']]

    ratio_1 = consistency_df['relative_time_online'].apply(lambda x: sum(x[int(len(x)/2):])/sum(x[:int(len(x)/2)]))
    ratio_1.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio_1.fillna(0, inplace=True)

    ratio_2 = consistency_df['relative_video_clicks'].apply(lambda x: sum(x[int(len(x)/2):])/sum(x[:int(len(x)/2)]))
    ratio_2.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio_2.fillna(0, inplace=True)    

    ratio_3 = consistency_df['relative_video_clicks'].apply(lambda x: sum(x[int(len(x)/2):])/sum(x[:int(len(x)/2)]))
    ratio_3.replace([np.inf, -np.inf], np.nan, inplace=True)
    ratio_3.fillna(0, inplace=True)   

    ratio = (ratio_1 + ratio_2 + ratio_3)/3 if add_mean_session else (ratio_1 + ratio_2)/2

    first_half, second_half = give_top_bot_percent(ratio, n)
    _, uniform = give_top_bot_percent(np.abs(ratio - 1), n)

    first_half = np.array(first_half.index)
    second_half = np.array(second_half.index)
    uniform = np.array(uniform.index)

    lim = np.min([len(first_half), len(uniform)])

    first_half = first_half[np.random.choice(len(first_half), size=(lim,), replace=False)] if len(first_half) > lim else first_half
    second_half = second_half[np.random.choice(len(second_half), size=(lim,), replace=False)] if len(second_half) > lim else second_half
    uniform = uniform[np.random.choice(len(uniform), size=(lim,), replace=False)] if len(uniform) > lim else uniform

    return first_half, second_half, uniform

def regularity(features_df, n):
    '''Regularity: Higher/Lower peaks'''
    # Top Intersection
    # Bot Union
    regularity_df = features_df[['user_id', 'regularity_periodicity_m1', 'regularity_periodicity_m2', 'regularity_periodicity_m3']]
    top_m1, bot_m1 = give_top_bot_percent(regularity_df['regularity_periodicity_m1'], n)
    top_m2, bot_m2 = give_top_bot_percent(regularity_df['regularity_periodicity_m2'], n)
    top_m3, bot_m3 = give_top_bot_percent(regularity_df['regularity_periodicity_m3'], n)
    
    top_intersection = np.array(list(set(top_m1.index).intersection(top_m2.index).intersection(top_m3.index)))
    bot_intersection = np.array(list(set(bot_m1.index).union(bot_m2.index).union(bot_m3.index)))
    
    lim = np.min([len(top_intersection), len(bot_intersection)])
    
    top_intersection = top_intersection[np.random.choice(len(top_intersection), size=(lim,), replace=False)] if len(top_intersection) > lim else top_intersection
    bot_intersection = bot_intersection[np.random.choice(len(bot_intersection), size=(lim,), replace=False)] if len(bot_intersection) > lim else bot_intersection
    
    return top_intersection, bot_intersection

def proactivity(features_df, n):
    '''Proactivity: Anticipated/Delayed'''
    # Top: Intersection
    # Bot: Union
    proactivity_df = features_df[['user_id', 'content_anticipation', 'delay_lecture']]
    
    top_anticipation, bot_anticipation = give_top_bot_percent(proactivity_df['content_anticipation'].apply(sum), n)
    top_delay, bot_delay = give_top_bot_percent(proactivity_df['delay_lecture'].apply(sum), n)
    
    top_intersection = np.array(list(set(top_anticipation.index).union(bot_delay.index)))
    bot_intersection = np.array(list(set(bot_anticipation.index).union(top_delay.index)))
    
    lim = np.min([len(top_intersection), len(bot_intersection)])
    
    top_intersection = top_intersection[np.random.choice(len(top_intersection), size=(lim,), replace=False)] if len(top_intersection) > lim else top_intersection
    bot_intersection = bot_intersection[np.random.choice(len(bot_intersection), size=(lim,), replace=False)] if len(bot_intersection) > lim else bot_intersection
    
    return top_intersection, bot_intersection

def control(features_df, n):
    '''Control: Higher/Lower intensity'''
    # Waiting for more features
    
    control_df = features_df[['user_id', 'frequency_action_Video.Pause', 'fraction_spent_ratio_duration_Video.Play', 'speed_playback_ mean']]
    
    top_pause_freq, bot_pause_freq = give_top_bot_percent(control_df['frequency_action_Video.Pause'].apply(sum), n)
    top_frac_spent, bot_frac_spent = give_top_bot_percent(control_df['fraction_spent_ratio_duration_Video.Play'].apply(sum), n)
    top_change_rate, bot_change_rate = give_top_bot_percent(control_df['speed_playback_ mean'].apply(sum), n)
    
    top_intersection = np.array(list(set(top_pause_freq.index).union(top_frac_spent.index).union(top_change_rate.index)))
    bot_intersection = np.array(list(set(bot_pause_freq.index).union(bot_frac_spent.index).union(bot_change_rate.index)))
    #top_intersection = np.array(list(set(top_frac_spent.index).intersection(top_change_rate.index)))
    #bot_intersection = np.array(list(set(bot_frac_spent.index).intersection(bot_change_rate.index)))
   
    lim = np.min([len(top_intersection), len(bot_intersection)])
    
    top_intersection = top_intersection[np.random.choice(len(top_intersection), size=(lim,), replace=False)] if len(top_intersection) > lim else top_intersection
    bot_intersection = bot_intersection[np.random.choice(len(bot_intersection), size=(lim,), replace=False)] if len(bot_intersection) > lim else bot_intersection
    
    return top_intersection, bot_intersection

def assessment(features_df, n):
    '''Assessment: Higher/Lower intensity'''
    # Top: Intersection
    # Bot: Union
    assessment_df = features_df[['user_id', 'competency_strength', 'student_shape']]
    
    top_competency_strength, bot_competency_strength = give_top_bot_percent(assessment_df['competency_strength'].apply(sum), n)
    top_student_shape, bot_student_shape = give_top_bot_percent(assessment_df['student_shape'].apply(sum), n)
    
    top_intersection = np.array(list(set(top_student_shape.index).intersection(top_competency_strength.index)))
    bot_intersection = np.array(list(set(bot_student_shape.index).union(bot_competency_strength.index)))
    
    lim = np.min([len(top_intersection), len(bot_intersection)])
    
    top_intersection = top_intersection[np.random.choice(len(top_intersection), size=(lim,), replace=False)] if len(top_intersection) > lim else top_intersection
    bot_intersection = bot_intersection[np.random.choice(len(bot_intersection), size=(lim,), replace=False)] if len(bot_intersection) > lim else bot_intersection
    
    return top_intersection, bot_intersection
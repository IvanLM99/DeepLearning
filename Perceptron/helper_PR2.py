#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def my_plot(w1, w2, bias, test_inputs, correct_outputs):
    """
    This function takes as input the parameters of a line: w1, w2, and bias.
    It also receives a set of points in 'test_inputs', an array of tuples,
    and a set of labels in 'test_outputs', an array of boolean elements.
    The output is a plot showing all points. Points are blue if their label
    is True and red if False. The line appears twice, once in blue and once
    in red, to differentiate the areas it separates. The blue area represents
    where points satisfy the equation w1*x1 + w2*x2 + bias > 0.
    """
    # Initialize arrays for red and blue points
    x_red, y_red, x_blue, y_blue = np.array(
        []), np.array(
        []), np.array(
            []), np.array(
                [])

    # Classify points into red and blue based on correct outputs
    for k, correct_output in enumerate(correct_outputs):
        if not correct_output:
            x_red = np.append(x_red, test_inputs[k][0])
            y_red = np.append(y_red, test_inputs[k][1])
        else:
            x_blue = np.append(x_blue, test_inputs[k][0])
            y_blue = np.append(y_blue, test_inputs[k][1])

    # Determine the position and slope of the line
    w2 = w2 if w2 != 0 else 0.001  # Avoid division by zero
    z = np.sign(w2)  # Positive zone above or below the line
    m = np.sign(-w1 / w2)  # Slope of the line
    m = m if m != 0 else 1

    # Calculate ranges for plotting
    x_min, x_max = min(np.append(x_red, x_blue)), max(np.append(x_red, x_blue))
    y_min, y_max = min(np.append(y_red, y_blue)), max(np.append(y_red, y_blue))
    rango_x, rango_y = x_max - x_min, y_max - y_min

    # Define points for drawing the line
    eps = 0.02 * max([rango_x, rango_y])  # Small offset
    x_val1 = np.array([x_min - 0.25 * rango_x, x_max + 0.25 * rango_x])
    x_val2 = x_val1 + eps * z * m
    y_val1 = -w1 / w2 * x_val1 - bias / w2
    y_val2 = y_val1 - eps * z

    # Plot formatting for points and lines
    fmt_red, fmt_blue = 'or', 'ob'
    fmt_val1, fmt_val2 = '-b', '-r'

    # Draw plot with points and lines
    plt.plot(x_red, y_red, fmt_red, linewidth=1, markersize=10)
    plt.plot(x_blue, y_blue, fmt_blue, linewidth=1, markersize=10)
    plt.plot(x_val1, y_val1, fmt_val1, linewidth=2, markersize=1)
    plt.plot(x_val2, y_val2, fmt_val2, linewidth=2, markersize=1)
    plt.ylim(y_min - 0.25 * rango_y, y_max + 0.25 * rango_y)
    plt.show()


def evaluate(weight1, weight2, bias, test_inputs,
             correct_outputs, extended=True):
    """
    This function evaluates the test inputs against the given weights and bias.
    It calculates the linear combination for each input, determines the output,
    and checks if it matches the correct output. It then prints the number of
    incorrect predictions and, if extended output is requested, displays a
    detailed dataframe. Finally, it calls my_plot to visualize the results.
    """
    outputs = []

    # Generate outputs for each test input
    for test_input, correct_output in zip(test_inputs, correct_outputs):
        linear_combination = weight1 * \
            test_input[0] + weight2 * test_input[1] + bias
        output = int(linear_combination >= 0)
        is_correct_string = 'Yes' if output == correct_output else 'No'
        outputs.append([test_input[0], test_input[1],
                       linear_combination, output, is_correct_string])

    # Calculate and print results
    num_wrong = len([output[4] for output in outputs if output[4] == 'No'])
    output_frame = pd.DataFrame(
        outputs,
        columns=[
            'Input 1',
            '  Input 2',
            '  Linear Combination',
            '  Activation Output',
            '  Is Correct'])
    if not num_wrong:
        print('Perfect!\n')
    else:
        print(f'You got {num_wrong} wrong.\n')

    # Print extended data if requested
    if extended:
        print(output_frame.to_string(index=False))

    # Visualize the plot
    my_plot(weight1, weight2, bias, test_inputs, correct_outputs)


def plot_dots(blue_points_calc, red_points_calc,
              blue_points_gt=None, red_points_gt=None):
    """
    This function plots points for visualization. It takes two sets of points:
    one calculated (blue_points_calc, red_points_calc) and one ground truth
    (blue_points_gt, red_points_gt). Points from the calculated set are
    plotted with smaller markers, and ground truth points with larger markers.
    """
    # Initialize ground truth points
    blue_points_gt = [] if blue_points_gt is None else blue_points_gt
    red_points_gt = [] if red_points_gt is None else red_points_gt

    # Initialize arrays for calculated and ground truth points
    x_blue_c, y_blue_c, x_red_c, y_red_c = np.array([]), np.array(
        []), np.array([]), np.array([])
    x_blue_gt, y_blue_gt, x_red_gt, y_red_gt = np.array([]), np.array(
        []), np.array([]), np.array([])

    # Extract coordinates for calculated blue and red points
    for (x, y) in blue_points_calc:
        x_blue_c = np.append(x_blue_c, x)
        y_blue_c = np.append(y_blue_c, y)
    for (x, y) in red_points_calc:
        x_red_c = np.append(x_red_c, x)
        y_red_c = np.append(y_red_c, y)

    # Extract coordinates for ground truth blue and red points
    for (x, y) in blue_points_gt:
        x_blue_gt = np.append(x_blue_gt, x)
        y_blue_gt = np.append(y_blue_gt, y)
    for (x, y) in red_points_gt:
        x_red_gt = np.append(x_red_gt, x)
        y_red_gt = np.append(y_red_gt, y)

    # Formatting for plotting
    fmt_blue, fmt_red = 'ob', 'or'

    # Draw plot with calculated and ground truth points
    plt.plot(x_blue_c, y_blue_c, fmt_blue, linewidth=1, markersize=6)
    plt.plot(x_red_c, y_red_c, fmt_red, linewidth=1, markersize=6)
    plt.plot(x_blue_gt, y_blue_gt, fmt_blue, linewidth=1, markersize=10)
    plt.plot(x_red_gt, y_red_gt, fmt_red, linewidth=1, markersize=10)
    plt.ylim(0.3, 1.2)
    plt.xlim(0.3, 1.2)
    plt.show()

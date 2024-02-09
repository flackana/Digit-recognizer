'''Here we find functions used in other notebooks'''
import matplotlib.pyplot as plt

def plot_a_digit(data_frame, i):
    """Plots any digit from the data set.

    Args:
        data_frame (DataFrame): Original data frame for x_train. 
        One row should have only 784 entries, each for one pixel.
        i (Int): the index of the digit we want to plot, 
        from 0 to len(DataFrame)
    """
    digit = data_frame.iloc[i, :].values
    digit_image = digit.reshape(28, 28)
    plt.imshow(digit_image, cmap="binary")
    plt.axis("off")
    plt.show()
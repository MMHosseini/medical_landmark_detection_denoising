from config import Constants, Variables
import tensorflow as tf


class CustomLoss:
    def calculate_mae(self, x_pr, x_gt):
        loss_mae = tf.reduce_mean(tf.abs(x_gt - x_pr))
        return loss_mae

    def calculate_mse(self, x_pr, x_gt):
        loss_mse = tf.math.reduce_mean(tf.square(x_gt - x_pr))
        return loss_mse


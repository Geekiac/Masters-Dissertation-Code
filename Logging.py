"""Consistent logging of messages across the scripts
"""

__all__ = ['log', 'LoggingWrapper']
__version__ = '0.1'
__author__ = "Steven Smith"

from time import time, asctime, strftime


def log(message):
    """Displays a logging message to the screen without wrapping a code block
    as LoggingWrapper does.

    :param message: The message to log to the screen along with the time
    """
    print("%s - %s" % (strftime('%H:%M:%S'), message))


class LoggingWrapper(object):
    """LoggingWrapper is to be used in a "with" block to log the start and end
    of a code block."""

    def __init__(self, message):
        """Initialises the Logging wrapper object with a message and a start
        time

        :param message: Message to be displayed at the start and the end of
        the block of code.
        """
        self.start = time()
        self.message = message

    def __enter__(self):
        """Logs a message at the start of the with block.

        :return: self, the current instance
        """
        log("%s : started" % self.message)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Logs a message at the end of the with block.

        :param exc_type: the exception type that caused the exit - None if no
                         exception occurred
        :param exc_value: the exception value that caused the exit - None if
                          no exception occurred
        :param traceback: the call stack information where the exception
                          occurred
        """
        end = time()
        log("%s : finished after %1.3f seconds" % (self.message,
                                                   end - self.start))

    def elapsed(self):
        """The time elapsed in this block

        :return: the elapsed time
        """
        return time() - self.start
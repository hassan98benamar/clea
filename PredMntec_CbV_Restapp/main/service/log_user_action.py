import logging


def user_action_logging(action_type, *args, logger_name='userAction'):
    """log user action

    Parameters
    ----------
    action_type : str
        type of action (predictbyid)

    args* : str list
        request argument

    logger_name : logging.logger
        logger object

    """
    nb_field = 8
    logger = logging.getLogger(logger_name)

    str_temp = action_type
    for arg in args:
        if type(arg) is list:
            str_temp = "{},{}".format(str_temp, '|'.join(arg))
        else:
            str_temp = "{},{}".format(str_temp, arg)
    current_nb_field  = str_temp.count(',') + 1
    if current_nb_field < nb_field:
        str_temp = str_temp + (',' * (nb_field - current_nb_field))
    elif current_nb_field > nb_field:
        logging.error("Too many field")
    logger.info(str_temp)



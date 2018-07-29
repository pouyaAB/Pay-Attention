"""
A set of config variables which will be used mostly in the constructors of the different classes.
This is a central config file for all the classes in the package.

'locals().update(config)' has been used to add these config variables as local variable in the other python files. 

"""

config = {}

config['image_size'] = 128
config['dataset_path'] = '/home/d3gan/development/datasets/record/sth_sth_' + str(config['image_size'])
config['sth_sth_path'] = '/home/d3gan/development/datasets/20BN_something_' + str(config['image_size'])
config['multi_object_dataset_path'] = '/home/d3gan/development/datasets/record/multi_object_' + str(config['image_size'])

config['cameras'] = ['camera-1']

config['record_path'] = '/home/d3gan/development/datasets/record/real_time'
config['robot_command_file'] = '/home/d3gan/development/datasets/record/real_time/commands.csv'
config['camera_topics'] = ['/camera1/usb_cam1/image_raw', '/camera2/usb_cam2/image_raw', '/camera3/usb_cam3/image_raw']
config['task'] = '5001'

config['tasks'] = ['5001', '5002', '5003', '5004']
config['batch_size'] = 20
config['sequence_size'] = 4
config['csv_col_num'] = 10
config['num_channels'] = 3
config['output_size'] = 7
config['hidden_dimension'] = 64
config['latent_size'] = 64
config['num_mixture'] = 32

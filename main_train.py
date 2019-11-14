import data_pipe
import networks_model
import custom_callback
import numpy as np
import datetime
import pickle
from keras import optimizers


'''make_generator_pipeline'''
train_crop_generator, test_crop_generator = data_pipe.crop_tiny_image_generator(batch_size=64, is_cluster=True)


### training full model

'''build model'''
model = networks_model.full_model(num_classes=200, input_shape=(56, 56, 3))     # crop image 64 -> 56

'''set optimizer and callback function (set training stop condition and save model having best acc)'''
sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
time_config_callback = custom_callback.time_config()

'''model training'''
start_time = datetime.datetime.now().strftime('%B-%d %H:%M:%S')
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
history = model.fit_generator(train_crop_generator, steps_per_epoch=1500, epochs=100, verbose=0,
                              validation_data=test_crop_generator, validation_steps=150, callbacks=[time_config_callback])
print('model fitting start:', start_time)
print('model fitting end and save:', datetime.datetime.now().strftime('%B-%d %H:%M:%S'))

with open('saved_model/clusterd_data/train_history/full_models_acc_histroy.sav', 'wb') as file:
    pickle.dump(history, file)
with open('saved_model/clusterd_data/train_history/full_models_time_history.sav', 'wb') as file:
    pickle.dump(time_config_callback, file)
model.save('saved_model/clusterd_data/full_model.h5')
print(time_config_callback.best_score)
print(model.evaluate_generator(test_crop_generator, steps=1000))


### training layer_reduced_model

'''build model'''
model = networks_model.layer_reduced_model(num_classes=200, input_shape=(56, 56, 3))     # crop image 64 -> 56

'''set optimizer and callback function (set training stop condition and save model having best acc)'''
sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
time_config_callback = custom_callback.time_config()

'''model training'''
start_time = datetime.datetime.now().strftime('%B-%d %H:%M:%S')
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
history = model.fit_generator(train_crop_generator, steps_per_epoch=1500, epochs=100, verbose=0,
                              validation_data=test_crop_generator, validation_steps=150, callbacks=[time_config_callback])
print('model fitting start:', start_time)
print('model fitting end and save:', datetime.datetime.now().strftime('%B-%d %H:%M:%S'))

with open('saved_model/clusterd_data/train_history/layer_reduced_model_acc_histroy.sav', 'wb') as file:
    pickle.dump(history, file)
with open('saved_model/clusterd_data/train_history/layer_reduced_model_time_history.sav', 'wb') as file:
    pickle.dump(time_config_callback, file)
model.save('saved_model/clusterd_data/layer_reduced_model.h5')
print(time_config_callback.best_score)
print(model.evaluate_generator(test_crop_generator, steps=1000))


### training filter_reduced_model

'''build model'''
model = networks_model.filter_reduced_model(num_classes=200, input_shape=(56, 56, 3))     # crop image 64 -> 56

'''set optimizer and callback function (set training stop condition and save model having best acc)'''
sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
time_config_callback = custom_callback.time_config()

'''model training'''
start_time = datetime.datetime.now().strftime('%B-%d %H:%M:%S')
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
history = model.fit_generator(train_crop_generator, steps_per_epoch=1500, epochs=100, verbose=0,
                              validation_data=test_crop_generator, validation_steps=150, callbacks=[time_config_callback])
print('model fitting start:', start_time)
print('model fitting end and save:', datetime.datetime.now().strftime('%B-%d %H:%M:%S'))

with open('saved_model/clusterd_data/train_history/filter_reduced_model_acc_histroy.sav', 'wb') as file:
    pickle.dump(history, file)
with open('saved_model/clusterd_data/train_history/filter_reduced_model_time_history.sav', 'wb') as file:
    pickle.dump(time_config_callback, file)
model.save('saved_model/clusterd_data/filter_reduced_model.h5')
print(time_config_callback.best_score)
print(model.evaluate_generator(test_crop_generator, steps=1000))


### training both_reduced_model

'''build model'''
model = networks_model.both_reduced_model(num_classes=200, input_shape=(56, 56, 3))     # crop image 64 -> 56

'''set optimizer and callback function (set training stop condition and save model having best acc)'''
sgd = optimizers.SGD(lr=0.01, decay=0, momentum=0.9, nesterov=True)
time_config_callback = custom_callback.time_config()

'''model training'''
start_time = datetime.datetime.now().strftime('%B-%d %H:%M:%S')
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
history = model.fit_generator(train_crop_generator, steps_per_epoch=1500, epochs=100, verbose=0,
                              validation_data=test_crop_generator, validation_steps=150, callbacks=[time_config_callback])
print('model fitting start:', start_time)
print('model fitting end and save:', datetime.datetime.now().strftime('%B-%d %H:%M:%S'))

with open('saved_model/clusterd_data/train_history/both_reduced_model_acc_histroy.sav', 'wb') as file:
    pickle.dump(history, file)
with open('saved_model/clusterd_data/train_history/both_reduced_model_time_history.sav', 'wb') as file:
    pickle.dump(time_config_callback, file)
model.save('saved_model/clusterd_data/both_reduced_model.h5')
print(time_config_callback.best_score)
print(model.evaluate_generator(test_crop_generator, steps=1000))

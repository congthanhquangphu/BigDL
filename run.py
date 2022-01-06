import time
from source_code import *  # Helper
# Boot training process
start = time.time()
trained_model = optimizer.optimize()
end = time.time()
time1 = end - start
print("Optimization Done.")
print( 'Train Time: ' + str(time1))


start = time.time()
predictions = trained_model.predict(test_data)
imshow(np.column_stack([np.array(s.features[0].to_ndarray()).reshape(28,28) for s in test_data.take(8)]),cmap='gray'); plt.axis('off')
print('Ground Truth labels:')
print(', '.join(str(map_groundtruth_label(s.label.to_ndarray())) for s in test_data.take(8)))
print('Predicted labels:')
print(', '.join(str(map_predict_label(s)) for s in predictions.take(8)))
end = time.time()
time1 = end - start
print( 'Predicted Time: ' + str(time1))
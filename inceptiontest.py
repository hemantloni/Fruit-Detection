from __future__ import print_function
from PIL import Image     
import os    
import numpy as np
import sys
import tensorflow as tf
import cv2
from PIL import ImageDraw
import mlptest

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util



def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    print("load_image_into_numpy_array")
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
    
def inceptionfunc(path):
    print("hello")  
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH ='ssd_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

            category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
   
    
    print("*****************main********************")

    with detection_graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                                                                                    
                                                                                    
            image = Image.open(path).convert("RGB")
            image = image.resize((120,120))
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.TEST_IMAGE_PATHS
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            
            # Visualization of the results of a detection.
            print("start croping")
            image_new = Image.fromarray(image_np)
            image_boundaries = output_dict['detection_boxes'][0]
            image_boundaries = image_boundaries.tolist()
            xmin, ymin, xmax, ymax = image_boundaries[0], image_boundaries[1], image_boundaries[2], image_boundaries[3]
            im_width, im_height = image_new.size
            (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width),
                                            int(ymin * im_height), int(ymax * im_height))
            cropped_image = image_new.crop((left, top, right, bottom))
            cropped_image.save('cropped_one.png')
            h,w=image_new.size
            # Create same size alpha layer with circle
            alpha = Image.new('L', image_new.size,0)
            draw = ImageDraw.Draw(alpha)
            draw.pieslice([55,55,75,75],0,360,fill=255)
            # Convert alpha Image to numpy array
            npAlpha=np.array(alpha)
            # Add alpha layer to RGB
            npImage=np.dstack((image_new,npAlpha))
            # Save with alpha
            new_path ='crop/'
            Image.fromarray(npImage).save(new_path+'{}'.format("my_crop.png"))
            
            #ima.save(new_path+'{}'.format("my_crop.JPG"))
            print("your pic is ready")  
            
            vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=2)
            cat_name = category_index[output_dict['detection_classes'][0]]['name']
    
            img = Image.fromarray(image_np, 'RGB')
            img.save("my_new.jpg") 
        
            pathtreturn="my_new.jpg"
            cat_id=[]
            # apple=.1,banana=.2,broccli=.3,carrot=.4,orange=.5,else=.6
            
            if cat_name == "apple":
                cat_id.append(".1")
                    
            elif cat_name == "banana":
                cat_id.append(".2")
                    
            elif cat_name == "broccoli":
                cat_id.append(".3")
                    
            elif  cat_name == "carrot":
                cat_id.append(".4")
                    
            elif  cat_name == "orange":
                 cat_id.append(".5") 
                
            else:
                cat_id.append(".6")
            
            calory=0
            croppath="crop/my_crop.png"
            #print(float(cat_id[0]))
            calory=mlptest.mlptest(croppath,float(cat_id[0]))
            calory=calory[0]
            if not calory:
                calory= 24.24
                
          
    return calory ,pathtreturn  ,cat_name   

        